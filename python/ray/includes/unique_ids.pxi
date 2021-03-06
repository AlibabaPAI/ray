"""This is a module for unique IDs in Ray.
We define different types for different IDs for type safety.

See https://github.com/ray-project/ray/issues/3721.
"""

# WARNING: Any additional ID types defined in this file must be added to the
# _ID_TYPES list at the bottom of this file.
import os

from ray.includes.unique_ids cimport (
    CActorCheckpointID,
    CActorClassID,
    CActorHandleID,
    CActorID,
    CClientID,
    CConfigID,
    CJobID,
    CFunctionID,
    CObjectID,
    CTaskID,
    CUniqueID,
    CWorkerID,
)

from ray.utils import decode


def check_id(b, size=kUniqueIDSize):
    if not isinstance(b, bytes):
        raise TypeError("Unsupported type: " + str(type(b)))
    if len(b) != size:
        raise ValueError("ID string needs to have length " +
                         str(size))


cdef extern from "ray/common/constants.h" nogil:
    cdef int64_t kUniqueIDSize
    cdef int64_t kMaxTaskPuts


cdef class BaseID:

    # To avoid the error of "Python int too large to convert to C ssize_t",
    # here `cdef size_t` is required.
    cdef size_t hash(self):
        pass

    def binary(self):
        pass

    def size(self):
        pass

    def hex(self):
        pass

    def is_nil(self):
        pass

    def __hash__(self):
        return self.hash()

    def __eq__(self, other):
        return type(self) == type(other) and self.binary() == other.binary()

    def __ne__(self, other):
        return self.binary() != other.binary()

    def __bytes__(self):
        return self.binary()

    def __hex__(self):
        return self.hex()

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.hex() + ")"

    def __str__(self):
        return self.__repr__()

    def __reduce__(self):
        return type(self), (self.binary(),)

    def redis_shard_hash(self):
        # NOTE: The hash function used here must match the one in
        # GetRedisContext in src/ray/gcs/tables.h. Changes to the
        # hash function should only be made through std::hash in
        # src/common/common.h.
        # Do not use __hash__ that returns signed uint64_t, which
        # is different from std::hash in c++ code.
        return self.hash()


cdef class UniqueID(BaseID):
    cdef CUniqueID data

    def __init__(self, id):
        check_id(id)
        self.data = CUniqueID.FromBinary(id)

    @classmethod
    def from_binary(cls, id_bytes):
        if not isinstance(id_bytes, bytes):
            raise TypeError("Expect bytes, got " + str(type(id_bytes)))
        return cls(id_bytes)

    @classmethod
    def nil(cls):
        return cls(CUniqueID.Nil().Binary())


    @classmethod
    def from_random(cls):
        return cls(os.urandom(CUniqueID.Size()))

    def size(self):
        return CUniqueID.Size()

    def binary(self):
        return self.data.Binary()

    def hex(self):
        return decode(self.data.Hex())

    def is_nil(self):
        return self.data.IsNil()

    cdef size_t hash(self):
        return self.data.Hash()


cdef class ObjectID(BaseID):
    cdef CObjectID data

    def __init__(self, id):
        check_id(id)
        self.data = CObjectID.FromBinary(<c_string>id)

    cdef CObjectID native(self):
        return <CObjectID>self.data

    def size(self):
        return CObjectID.Size()

    def binary(self):
        return self.data.Binary()

    def hex(self):
        return decode(self.data.Hex())

    def is_nil(self):
        return self.data.IsNil()

    cdef size_t hash(self):
        return self.data.Hash()

    @classmethod
    def nil(cls):
        return cls(CObjectID.Nil().Binary())

    @classmethod
    def from_random(cls):
        return cls(os.urandom(CObjectID.Size()))


cdef class TaskID(BaseID):
    cdef CTaskID data

    def __init__(self, id):
        check_id(id, CTaskID.Size())
        self.data = CTaskID.FromBinary(<c_string>id)

    cdef CTaskID native(self):
        return <CTaskID>self.data

    def size(self):
        return CTaskID.Size()

    def binary(self):
        return self.data.Binary()

    def hex(self):
        return decode(self.data.Hex())

    def is_nil(self):
        return self.data.IsNil()

    cdef size_t hash(self):
        return self.data.Hash()

    @classmethod
    def nil(cls):
        return cls(CTaskID.Nil().Binary())

    @classmethod
    def size(cla):
        return CTaskID.Size()

    @classmethod
    def from_random(cls):
        return cls(os.urandom(CTaskID.Size()))


cdef class ClientID(UniqueID):

    def __init__(self, id):
        check_id(id)
        self.data = CClientID.FromBinary(<c_string>id)

    cdef CClientID native(self):
        return <CClientID>self.data


cdef class JobID(UniqueID):

    def __init__(self, id):
        check_id(id)
        self.data = CJobID.FromBinary(<c_string>id)

    cdef CJobID native(self):
        return <CJobID>self.data

cdef class WorkerID(UniqueID):

    def __init__(self, id):
        check_id(id)
        self.data = CWorkerID.FromBinary(<c_string>id)

    cdef CWorkerID native(self):
        return <CWorkerID>self.data

cdef class ActorID(UniqueID):

    def __init__(self, id):
        check_id(id)
        self.data = CActorID.FromBinary(<c_string>id)

    cdef CActorID native(self):
        return <CActorID>self.data


cdef class ActorHandleID(UniqueID):

    def __init__(self, id):
        check_id(id)
        self.data = CActorHandleID.FromBinary(<c_string>id)

    cdef CActorHandleID native(self):
        return <CActorHandleID>self.data


cdef class ActorCheckpointID(UniqueID):

    def __init__(self, id):
        check_id(id)
        self.data = CActorCheckpointID.FromBinary(<c_string>id)

    cdef CActorCheckpointID native(self):
        return <CActorCheckpointID>self.data


cdef class FunctionID(UniqueID):

    def __init__(self, id):
        check_id(id)
        self.data = CFunctionID.FromBinary(<c_string>id)

    cdef CFunctionID native(self):
        return <CFunctionID>self.data


cdef class ActorClassID(UniqueID):

    def __init__(self, id):
        check_id(id)
        self.data = CActorClassID.FromBinary(<c_string>id)

    cdef CActorClassID native(self):
        return <CActorClassID>self.data

_ID_TYPES = [
    ActorCheckpointID,
    ActorClassID,
    ActorHandleID,
    ActorID,
    ClientID,
    JobID,
    WorkerID,
    FunctionID,
    ObjectID,
    TaskID,
    UniqueID,
]

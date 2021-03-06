#ifndef RAY_CORE_WORKER_TASK_EXECUTION_H
#define RAY_CORE_WORKER_TASK_EXECUTION_H

#include "ray/common/buffer.h"
#include "ray/common/status.h"
#include "ray/core_worker/common.h"
#include "ray/core_worker/context.h"
#include "ray/core_worker/object_interface.h"
#include "ray/core_worker/transport/transport.h"
#include "ray/rpc/client_call.h"
#include "ray/rpc/worker/worker_client.h"
#include "ray/rpc/worker/worker_server.h"

namespace ray {

class CoreWorker;

namespace raylet {
class TaskSpecification;
}

/// The interface that contains all `CoreWorker` methods that are related to task
/// execution.
class CoreWorkerTaskExecutionInterface {
 public:
  CoreWorkerTaskExecutionInterface(WorkerContext &worker_context,
                                   std::unique_ptr<RayletClient> &raylet_client,
                                   CoreWorkerObjectInterface &object_interface);

  /// The callback provided app-language workers that executes tasks.
  ///
  /// \param ray_function[in] Information about the function to execute.
  /// \param args[in] Arguments of the task.
  /// \return Status.
  using TaskExecutor =
      std::function<Status(const RayFunction &ray_function,
                           const std::vector<std::shared_ptr<RayObject>> &args,
                           const TaskInfo &task_info, int num_returns)>;

  /// Start receving and executes tasks in a infinite loop.
  /// \return Status.
  Status Run(const TaskExecutor &executor);

 private:
  /// Build arguments for task executor. This would loop through all the arguments
  /// in task spec, and for each of them that's passed by reference (ObjectID),
  /// fetch its content from store and; for arguments that are passed by value,
  /// just copy their content.
  ///
  /// \param spec[in] Task specification.
  /// \param args[out] The arguments for passing to task executor.
  ///
  Status BuildArgsForExecutor(const raylet::TaskSpecification &spec,
                              std::vector<std::shared_ptr<RayObject>> *args);

  /// Reference to the parent CoreWorker's context.
  WorkerContext &worker_context_;
  /// Reference to the parent CoreWorker's objects interface.
  CoreWorkerObjectInterface &object_interface_;

  /// All the task task receivers supported.
  std::unordered_map<int, std::unique_ptr<CoreWorkerTaskReceiver>> task_receivers_;

  /// The RPC server.
  rpc::GrpcServer worker_server_;

  /// Event loop where tasks are processed.
  boost::asio::io_service main_service_;

  /// The asio work to keep main_service_ alive.
  boost::asio::io_service::work main_work_;

  friend class CoreWorker;
};

}  // namespace ray

#endif  // RAY_CORE_WORKER_TASK_EXECUTION_H

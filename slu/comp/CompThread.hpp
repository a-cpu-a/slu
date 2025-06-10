/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <thread>
#include <variant>
#include <slu/ext/Mtx.hpp>

#include <slu/lang/BasicState.hpp>

#include <slu/comp/CompCfg.hpp>
#include <slu/comp/HandleTask.hpp>

namespace slu::comp
{


	inline void poolThread(const CompCfg& cfg,
		std::atomic_bool& shouldExit,
		std::condition_variable& cv,
		std::condition_variable& cvMain,
		Mutex<size_t>& tasksLeft,
		Mutex<std::vector<CompTask>>& tasks
	)
	{
		uint32_t lastTask = 0;
		TaskHandleStateMlir mlirState{
			.mc=mlir::MLIRContext(mlir::MLIRContext::Threading::DISABLED),
			.llvmCtx=llvm::LLVMContext() 
		};
		mlirState.mc.getOrLoadDialect<mlir::memref::MemRefDialect>();
		mlirState.mc.getOrLoadDialect<mlir::func::FuncDialect>();
		mlirState.mc.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
		mlirState.mc.getOrLoadDialect<mlir::arith::ArithDialect>();
		mlirState.mc.getOrLoadDialect<mlir::index::IndexDialect>();
		TaskHandleState state{
			.s = &mlirState,
			.target = mlir::LLVMConversionTarget{mlirState.mc},
			.typeConverter = mlir::LLVMTypeConverter{&mlirState.mc}
		};
		state.target.addLegalOp<mlir::ModuleOp>();

		mlir::RewritePatternSet patterns{ &mlirState.mc };

		mlir::populateAffineToStdConversionPatterns(patterns);
		mlir::populateSCFToControlFlowConversionPatterns(patterns);
		mlir::arith::populateArithToLLVMConversionPatterns(state.typeConverter, patterns);
		mlir::populateFinalizeMemRefToLLVMConversionPatterns(state.typeConverter, patterns);
		mlir::cf::populateControlFlowToLLVMConversionPatterns(state.typeConverter, patterns);
		mlir::populateFuncToLLVMConversionPatterns(state.typeConverter, patterns);

		state.s->patterns = mlir::FrozenRewritePatternSet{ std::move(patterns) };

		while (true)
		{
			std::unique_lock taskLock(tasks.lock);
			if (shouldExit)return;
			if (tasks.v.empty())
				cv.wait(taskLock, [&shouldExit, &tasks]() { return shouldExit || !tasks.v.empty(); });
			if (shouldExit)return;
			if (tasks.v.empty())continue;

			CompTask& taskRef = tasks.v.back();
			if (taskRef.taskId == lastTask)
				continue;//already did it.
			lastTask = taskRef.taskId;
			taskRef.threadsLeft--;
			bool isCompleterThread = taskRef.threadsLeft == 0;

			std::optional<CompTaskData> stackCopy;

			if (isCompleterThread)
			{
				stackCopy = std::move(taskRef.data);
				tasks.v.pop_back();
				taskLock.unlock();//only thread with this stuff, so no need to keep a lock!
			}
			CompTaskData& task = stackCopy.has_value()
				? stackCopy.value()
				: taskRef.data;//else, it was not invalidated

			//Complete it...
			handleTask(cfg, shouldExit, state, stackCopy.has_value() ? nullptr: &taskLock, task);

			if (isCompleterThread)
			{
				std::lock_guard _(tasksLeft.lock);
				_ASSERT(tasksLeft.v != 0);
				if (--tasksLeft.v == 0)
				{
					//wake up, all tasks are done!
					cvMain.notify_all();
				}
			}
		}
	}
}
module;
/*
** See Copyright Notice inside Include.hpp
*/

//#include <slu/comp/mlir/Pch.hpp>
export module slu.comp.comp_thread;

import a_cpu_a.mtx;
import slu.ast.op_info;
import slu.comp.cfg;
import slu.comp.handle_task;
import slu.lang.basic_state;

namespace slu::comp
{
	export void poolThread(const CompCfg& cfg,
		std::atomic_bool& shouldExit,
		std::condition_variable& cv,
		std::condition_variable& cvMain,
		a_cpu_a::Mutex<size_t>& tasksLeft,
		a_cpu_a::Mutex<std::vector<CompTask>>& tasks
	)
	{
		uint32_t lastTask = 0;
		TaskHandleStateMlir mlirState{
			.mc = mlir::MLIRContext(mlir::MLIRContext::Threading::DISABLED),
			.llvmCtx = llvm::LLVMContext()
		};
		mlirState.mc.getOrLoadDialect<mlir::memref::MemRefDialect>();
		mlirState.mc.getOrLoadDialect<mlir::func::FuncDialect>();
		mlirState.mc.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
		mlirState.mc.getOrLoadDialect<mlir::arith::ArithDialect>();
		mlirState.mc.getOrLoadDialect<mlir::index::IndexDialect>();
		mlirState.mc.getOrLoadDialect<mlir::DLTIDialect>();
		mlirState.mc.getOrLoadDialect<slu_dial::SluDialect>();


		//mlir::LowerToLLVMOptions llvmOptions(&mlirState.mc);
		//llvmOptions.overrideIndexBitwidth(64);

		TaskHandleState state{
			.s = &mlirState,
			.target = mlir::LLVMConversionTarget{mlirState.mc},
			.tyConv = mlir::LLVMTypeConverter{&mlirState.mc},
			.indexSize = 64
		};
		state.target.addLegalOp<mlir::ModuleOp>(); 

		//mlir::RewritePatternSet patterns{ &mlirState.mc };

		//mlir::populateAffineToStdConversionPatterns(patterns);
		//mlir::populateSCFToControlFlowConversionPatterns(patterns);
		//mlir::populateFinalizeMemRefToLLVMConversionPatterns(state.typeConverter, patterns);
		//mlir::cf::populateControlFlowToLLVMConversionPatterns(state.typeConverter, patterns);
		//mlir::index::populateIndexToLLVMConversionPatterns(state.typeConverter, patterns);
		//mlir::arith::populateArithToLLVMConversionPatterns(state.typeConverter, patterns);
		//mlir::populateFuncToLLVMConversionPatterns(state.typeConverter, patterns);


		//state.s->patterns = mlir::FrozenRewritePatternSet{ std::move(patterns) };

		mlirState.indexLay = mlir::DataLayoutEntryAttr::get(
			mlirState.opBuilder.getIndexType(),
			mlirState.opBuilder.getI32IntegerAttr(state.indexSize)
		);

		// Optional: enable IR printing/debugging
		mlirState.pm.enableVerifier(/*verifyPasses=*/true);
		//mlirState.pm.enableIRPrinting();

		// Add passes in order
		mlirState.pm.addPass(mlir::memref::createExpandStridedMetadataPass());	// memref → memref + affine + arith
		mlirState.pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());	// memref → LLVM
		mlirState.pm.addPass(mlir::createLowerAffinePass());					// Affine → SCF + arith
		mlirState.pm.addPass(mlir::createSCFToControlFlowPass());				// SCF → CF
		mlirState.pm.addPass(mlir::createConvertControlFlowToLLVMPass());		// cf → LLVM
		mlirState.pm.addPass(mlir::createArithToLLVMConversionPass());			// arith → LLVM
		mlirState.pm.addPass(mlir::createConvertIndexToLLVMPass());				// index → LLVM
		mlirState.pm.addPass(mlir::createConvertFuncToLLVMPass());				// func → LLVM

		// Optional: cleanup/canonicalization
		mlirState.pm.addPass(mlir::createCanonicalizerPass());
		//mlirState.pm.addPass(mlir::createSCCPPass());
		mlirState.pm.addPass(mlir::createCSEPass());
		//mlirState.pm.addPass(mlir::createSymbolDCEPass());

		mlir::registerBuiltinDialectTranslation(mlirState.mc);
		mlir::registerLLVMDialectTranslation(mlirState.mc);

		{
			lang::ModPath mp{ "std","ops" };
			parse::BasicMpDb{ &state.mpDb }.get<false>(mp);
			for (auto& i : ast::unOpTraitNames)
			{
				mp.emplace_back(i);
				parse::BasicMpDb{ &state.mpDb }.get<false>(mp);
				mp.pop_back();
			}
			for (auto& i : ast::binOpTraitNames)
			{
				mp.emplace_back(i);
				parse::BasicMpDb{ &state.mpDb }.get<false>(mp);
				mp.pop_back();
			}
			for (auto& i : ast::postUnOpTraitNames)
			{
				mp.emplace_back(i);
				parse::BasicMpDb{ &state.mpDb }.get<false>(mp);
				mp.pop_back();
			}
		}

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
			handleTask(cfg, shouldExit, state, stackCopy.has_value() ? nullptr : &taskLock, task);

			if (isCompleterThread)
			{
				std::lock_guard _(tasksLeft.lock);
				Slu_assert(tasksLeft.v != 0);
				if (--tasksLeft.v == 0)
				{
					//wake up, all tasks are done!
					cvMain.notify_all();
				}
			}
		}
	}
}
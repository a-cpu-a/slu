/*
    A program file.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This file is part of Slu-c.

    Slu-c is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Slu-c is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with Slu-c.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;

//#include <slu/comp/mlir/Pch.hpp>
export module slu.comp.compile;

import a_cpu_a.mtx;
import slu.ast.mp_data;
import slu.comp.cfg;
import slu.comp.comp_thread;
import slu.comp.handle_task;
import slu.lang.basic_state;

namespace slu::comp
{
	export struct CodeGenEntrypoint
	{
		//path to file that defined this entry-point, or empty
		std::string entryPointFile;
		std::string fileName;
	};
	//Could repr some js file, some wasm blob, a jar / class, or
	// even some exe / dll.
	export struct CompEntryPoint : CodeGenEntrypoint
	{
		std::vector<uint8_t> contents;
	};
	export struct CompOutput
	{
		std::vector<CompEntryPoint> entryPoints;
		//TODO: info for lock file appending?
		//TODO: info for build cache files?
	};

	inline void waitForTasksToComplete(
	    a_cpu_a::Mutex<size_t>& tasksLeft, std::condition_variable& cvMain)
	{
		std::unique_lock tasksLeftLock(tasksLeft.lock);
		if (tasksLeft.v != 0)
			cvMain.wait(
			    tasksLeftLock, [&tasksLeft] { return tasksLeft.v == 0; });
	}
	inline void submitTask(const CompCfg& cfg, uint32_t taskId,
	    a_cpu_a::Mutex<std::vector<CompTask>>& tasks,
	    a_cpu_a::Mutex<size_t>& tasksLeft, std::condition_variable& cv,
	    std::condition_variable& cvMain,

	    CompTaskData&& data)
	{
		CompTask task;
		task.taskId = taskId;
		task.threadsLeft = 1;
		// Move a chunk into a new vector
		task.data = std::move(data);

		{
			std::lock_guard _(tasksLeft.lock);
			tasksLeft.v++;
		}
		{
			std::unique_lock _(tasks.lock);
			tasks.v.emplace_back(std::move(task));
		}
		cv.notify_one();
	}
	inline void submitConsensusTask(const CompCfg& cfg, uint32_t taskId,
	    a_cpu_a::Mutex<std::vector<CompTask>>& tasks,
	    a_cpu_a::Mutex<size_t>& tasksLeft, std::condition_variable& cv,
	    std::condition_variable& cvMain,

	    CompTaskData&& data)
	{
		CompTask task;
		task.taskId = taskId;
		task.threadsLeft = cfg.extraThreadCount;
		// Move a chunk into a new vector
		task.data = std::move(data);

		{
			std::lock_guard _(tasksLeft.lock);
			tasksLeft.v++;
		}
		{
			std::unique_lock _(tasks.lock);
			tasks.v.emplace_back(std::move(task));
		}
		cv.notify_all();

		waitForTasksToComplete(tasksLeft, cvMain);
	}

	export CompOutput compile(const CompCfg& cfg)
	{
		llvm::InitializeAllTargets();
		llvm::InitializeAllTargetMCs();
		llvm::InitializeAllAsmParsers();
		llvm::InitializeAllAsmPrinters();

		uint32_t nextTaskId = 1;
		a_cpu_a::Mutex<std::vector<CompTask>> tasks;
		a_cpu_a::Mutex<size_t> tasksLeft = 0;

		std::condition_variable cv;
		std::condition_variable cvMain; //Uses tasksLeft.lock!!
		std::atomic_bool shouldExit = false;

		std::vector<std::thread> pool;
		for (size_t i = 0; i < cfg.extraThreadCount; i++)
		{
			pool.emplace_back(
			    std::thread(poolThread, cfg, std::ref(shouldExit), std::ref(cv),
			        std::ref(cvMain), std::ref(tasksLeft), std::ref(tasks)));
		}
		{
			// Create a list of all files to compile
			std::vector<SluFile> sluFiles;
			sluFiles.reserve(cfg.rootPaths.size() * 10);
			for (std::string_view i : cfg.rootPaths)
			{
				auto list = cfg.getFileListRecPtr(i);
				for (std::string& file : list)
				{
					if (cfg.isFolderPtr(file))
						continue;
					if (!file.ends_with(".slu"))
						continue;
					auto content = cfg.getFileContentsPtr(file);
					if (!content.has_value())
						continue;

					sluFiles.emplace_back(
					    i, std::move(file), std::move(content.value()));
				}
			}

			if (cfg.extraThreadCount != 0)
			{
				size_t total = sluFiles.size();
				size_t filesPerThread = total     // *4, because we want to use
				    / (cfg.extraThreadCount * 4); // the task system a bit more.


				// If there arent enough files for all
				// the threads, just use 1 thread ig
				if (filesPerThread == 0)
					filesPerThread = 1;

				size_t baseSize = total / filesPerThread;
				size_t remainder = total % filesPerThread;

				auto it = std::make_move_iterator(sluFiles.begin());
				for (size_t i = 0; i < filesPerThread; ++i)
				{
					size_t chunkSize = baseSize + (i < remainder ? 1 : 0);


					CompTask res;
					res.taskId = nextTaskId++;
					res.threadsLeft = 1;
					// Move a chunk into a new vector
					res.data = CompTaskType::ParseFiles(it, it + chunkSize);
					it += chunkSize;

					{
						std::lock_guard _(tasksLeft.lock);
						tasksLeft.v++;
					}
					{
						std::unique_lock _(tasks.lock);
						tasks.v.emplace_back(std::move(res));
					}
					cv.notify_one();
				}

				waitForTasksToComplete(tasksLeft, cvMain);
				// All files parsed!
			} else
			{
				// TODO: Parse the files, if extraThreads==0
			}
		}
		Slu_assertOp(tasksLeft.v, ==, 0);

		a_cpu_a::RwLock<parse::BasicMpDbData> sharedDb;

		// unify asts using sharedDb
		submitConsensusTask(cfg, nextTaskId++, tasks, tasksLeft, cv, cvMain,
		    CompTaskType::ConsensusUnifyAsts{
		        .sharedDb = &sharedDb, .firstToArive = true});
		MergeAstsMap astMap;
		submitConsensusTask(cfg, nextTaskId++, tasks, tasksLeft, cv, cvMain,
		    CompTaskType::ConsensusMergeAsts{&astMap});
		//TODO: build some kind of dep graph?

		//TODO: + comptime eval (jit?)
		//type-inference/checking
		for (auto& [mp, file] : astMap)
		{ // Codegen on all the threads.
			submitTask(cfg, nextTaskId++, tasks, tasksLeft, cv, cvMain,
			    CompTaskType::TypeInfCheck{
			        MutFileStatList{std::span{file.pf.code}, file.path}
            });
		}
		waitForTasksToComplete(tasksLeft, cvMain);

		std::vector<CodeGenEntrypoint> eps;
		for (auto& [mp, file] : astMap)
		{
			if (file.path.ends_with("/main.slu"))
				eps.push_back(
				    {.entryPointFile = file.path, .fileName = "main.exe"});
		}
		for (auto& [mp, file] : astMap)
		{ // Codegen on all the threads.
			submitTask(cfg, nextTaskId++, tasks, tasksLeft, cv, cvMain,
			    CompTaskType::DoCodeGen{.statements
			        = {FileStatList{std::span{file.pf.code}, file.path}},
			        .entrypointId = uint32_t(eps.size() - 1)});
		}
		waitForTasksToComplete(tasksLeft, cvMain);

		CompOutput ret;
		{ // Merge code gen outputs
			GenCodeMap mergeOut;
			mergeOut.reserve(eps.size());

			submitConsensusTask(cfg, nextTaskId++, tasks, tasksLeft, cv, cvMain,
			    CompTaskType::ConsensusMergeGenCode(&mergeOut));
			uint32_t i = 0;
			for (auto& epInfo : eps)
			{
				auto& mergeOutItem = mergeOut[i++];
				CompEntryPoint epPdb{epInfo};
				epPdb.fileName += ".pdb";
				CompEntryPoint ep{std::move(epInfo)};

				lld::Result res;
				{
					std::vector<TmpFile> tmpFiles;
					for (auto& v : mergeOutItem)
					{
						auto sp = std::span{v};
						tmpFiles.emplace_back(cfg.mkTmpFilePtr(
						    std::bit_cast<std::span<const uint8_t>>(
						        sp) // likely a bit cursed!
						    ));
					}
					std::vector<std::string> strArgs = {"lld-link"};
					for (auto& f : tmpFiles)
						strArgs.push_back(f.realPath);
					strArgs.push_back("kernel32.lib");
					strArgs.push_back("libucrt.lib");
					strArgs.push_back("libcmt.lib");
					strArgs.push_back("/debug");
					;
					auto pdbFile = cfg.mkTmpFilePtr({});
					strArgs.push_back("/pdb:" + pdbFile.realPath);
					strArgs.push_back("/subsystem:console");
					auto outFile = cfg.mkTmpFilePtr({});
					strArgs.push_back("/out:" + outFile.realPath);

					std::vector<const char*> cstrArgs;
					cstrArgs.reserve(strArgs.size());
					for (auto& ij : strArgs)
						cstrArgs.push_back(ij.c_str());

					const lld::DriverDef driver
					    = {lld::Flavor::WinLink, &lld::coff::link};

					//TODO: custom outs/errs
					res = lld::lldMain(
					    cstrArgs, llvm::outs(), llvm::errs(), {driver});
					if (res.retCode != 0)
						cfg.errPtr("Linker failed to run, error: "
						    + std::to_string(res.retCode));
					if (!res.canRunAgain)
						return ret;
					//Read from out files
					auto optFile = cfg.getFileContentsPtr(pdbFile.realPath);
					if (optFile.has_value())
						epPdb.contents = std::move(*optFile);
					else
						cfg.logPtr("Linker didnt generate a pdb!");
					//
					optFile = cfg.getFileContentsPtr(outFile.realPath);
					if (optFile.has_value())
						ep.contents = std::move(*optFile);
					else
						cfg.errPtr("Linker didnt generate a output!");
				}
				ret.entryPoints.emplace_back(std::move(epPdb));
				ret.entryPoints.emplace_back(std::move(ep));
			}
		}
		shouldExit = true;
		cv.notify_all();
		for (auto& i : pool)
			i.join();

		return ret;
	}
} //namespace slu::comp
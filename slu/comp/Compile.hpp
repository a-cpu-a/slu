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

#include <slu/comp/CompThread.hpp>
#include <slu/comp/CompInclude.hpp>

namespace slu::comp
{
	inline void waitForTasksToComplete(Mutex<size_t>& tasksLeft, std::condition_variable& cvMain)
	{
		std::unique_lock tasksLeftLock(tasksLeft.lock);
		if (tasksLeft.v != 0)
			cvMain.wait(tasksLeftLock, [&tasksLeft] {return tasksLeft.v == 0; });
	}
	inline void submitTask(const CompCfg& cfg,
		uint32_t taskId,
		Mutex<std::vector<CompTask>>& tasks,
		Mutex<size_t>& tasksLeft, 
		std::condition_variable& cv, 
		std::condition_variable& cvMain,

		CompTaskData&& data
	)
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
	inline void submitConsensusTask(const CompCfg& cfg,
		uint32_t taskId,
		Mutex<std::vector<CompTask>>& tasks,
		Mutex<size_t>& tasksLeft, 
		std::condition_variable& cv, 
		std::condition_variable& cvMain,

		CompTaskData&& data
	)
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

	inline CompOutput compile(const CompCfg& cfg)
	{
		uint32_t nextTaskId = 1;
		Mutex<std::vector<CompTask>> tasks;
		Mutex<size_t> tasksLeft = 0;
		
		std::condition_variable cv;
		std::condition_variable cvMain;//Uses tasksLeft.lock!!
		std::atomic_bool shouldExit = false;

		std::vector<std::thread> pool;
		for (size_t i = 0; i < cfg.extraThreadCount; i++)
		{
			pool.emplace_back(std::thread(poolThread,cfg,
				std::ref(shouldExit),
				std::ref(cv),std::ref(cvMain),
				std::ref(tasksLeft),std::ref(tasks))
			);
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
					if (cfg.isFolderPtr(file)) continue;
					if (!file.ends_with(".slu")) continue;
					auto content = cfg.getFileContentsPtr(file);
					if (!content.has_value())continue;

					sluFiles.emplace_back(i,std::move(file), std::move(content.value()));
				}
			}

			if(cfg.extraThreadCount!=0)
			{
				size_t total = sluFiles.size();
				size_t filesPerThread = total / (cfg.extraThreadCount*4);// *4, because we want to use the task system a bit more.

				// If there arent enough files for all the threads, just use 1 thread ig
				if (filesPerThread == 0) filesPerThread = 1;

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
			}
			else
			{
				// TODO: Parse the files, if extraThreads==0
			}
		}
		_ASSERT(tasksLeft.v == 0);

		RwLock<parse::BasicMpDbData> sharedDb;

		// unify asts using sharedDb
		submitConsensusTask(cfg, nextTaskId++, tasks, tasksLeft, cv, cvMain, 
			CompTaskType::ConsensusUnifyAsts{
				.sharedDb = &sharedDb,
				.firstToArive = true
			}
		);
		MergeAstsMap astMap;
		submitConsensusTask(cfg, nextTaskId++, tasks, tasksLeft, cv, cvMain,
			CompTaskType::ConsensusMergeAsts{ &astMap }
		);
		//TODO: collect ast's
		//TODO: convert into something more? -> seperate global stats from the func code
		//TODO: build some kind of dep graph.
		//TODO: type-inference/checking + comptime eval + basic codegen to lua

		std::vector<CodeGenEntrypoint> eps;

		//TODO: run midlevel / optimize using all the threads.
		for (auto& [mp,file] : astMap)
		{// Codegen on all the threads.
			if (file.path.ends_with("/main.slu"))
			{
				eps.push_back({ .entryPointFile = file.path, .fileName = "main.exe" });
				submitTask(cfg,nextTaskId++,tasks,tasksLeft,cv,cvMain,
					CompTaskType::DoCodeGen{
						.statements = {std::span{file.pf.code.statList}},
						.entrypointId=uint32_t(eps.size() - 1)
				});
			}
		}
		waitForTasksToComplete(tasksLeft, cvMain);

		CompOutput ret;

		{ // Merge code gen outputs
			GenCodeMap mergeOut;
			mergeOut.reserve(eps.size());

			submitConsensusTask(cfg, nextTaskId++, tasks, tasksLeft, cv, cvMain,
				CompTaskType::ConsensusMergeGenCode(&mergeOut)
			);

			uint32_t i = 0;
			for (auto& epInfo : eps)
			{
				auto& mergeOutItem = mergeOut[i++];
				CompEntryPoint ep{std::move(epInfo)};
				
				size_t totalSize = 0;
				for (auto& j : mergeOutItem)
					totalSize += j.size();
				ep.contents.reserve(totalSize);
				for (auto& j : mergeOutItem)
				{
					ep.contents.insert(ep.contents.end(), j.begin(), j.end());
					j.clear();//dealloc now, to keep mem usage lower
				}
				mergeOutItem.clear();//it wont be used anymore

				ret.entryPoints.emplace_back(std::move(ep));
			}
		}

		shouldExit = true;
		cv.notify_all();
		for (auto& i : pool)
			i.join();

		return ret;
	}
}
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

namespace slu::comp
{
	//Could repr some js file, some wasm blob, a jar / class, or even some exe / dll.
	struct CompEntryPoint
	{
		std::string entryPointFile;//path to file that defined this entry-point, or empty
		std::string fileName;
		std::vector<uint8_t> contents;
	};
	struct CompOutput
	{
		std::vector<CompEntryPoint> entryPoints;
		//TODO: info for lock file appending?
		//TODO: info for build cache files?
	};

	inline CompOutput compile(const CompCfg& cfg)
	{
		uint32_t nextTaskId = 1;
		Mutex<std::vector<CompTask>> tasks;
		Mutex<size_t> tasksLeft;

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
				size_t filesPerThread = total / cfg.extraThreadCount;

				// If there arent enough files for all the threads, just use 1 thread ig
				if (filesPerThread == 0) filesPerThread = 1;

				size_t baseSize = total / filesPerThread;
				size_t remainder = total % filesPerThread;

				auto it = std::make_move_iterator(sluFiles.begin());
				for (size_t i = 0; i < filesPerThread; ++i)
				{
					size_t chunkSize = baseSize + (i < remainder ? 1 : 0);


					CompTask res;
					res.leaveForMain = false;
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

				{
					std::unique_lock tasksLeftLock(tasksLeft.lock);
					if (tasksLeft.v != 0)
						cvMain.wait(tasksLeftLock, [&tasksLeft] {return tasksLeft.v == 0; });
					// All files parsed!
				}
			}
			else
			{
				// TODO: Parse the files, if extraThreads==0
			}
		}


		shouldExit = true;
		cv.notify_all();
		for (auto& i : pool)
			i.join();
	}
}
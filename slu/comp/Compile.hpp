/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <thread>
#include <variant>

#include <slu/lang/BasicState.hpp>
#include <slu/ext/Mtx.hpp>

namespace slu::comp
{
	//
	// All views / spans are only required to live as long as the call.
	// The compiler wont emit `../`, so there is no need to sanitize them for that (except by you in rootPaths)
	// Your paths must not contain `\`
	// If one of your callbacks is thread-unsafe, add some mutex, because all of them will be called by many threads.
	//

	

	//TODO: implement compiler plugins, maybe redesign the api too.

	//Used to implement external namespaces (lua::ext::*)
	struct CompPlugin
	{
		// If it would need to call to some dll, or external code, it is likely not safe
		bool comptimeSafe : 1 = false;

		lang::ModPathView root;

		//TODO: choose a type for the list (string?) (mod path?)
		using GetItemListFn = std::vector<int>(*)(const lang::ModPathView& path);
		GetItemListFn getItemListPtr;

		using ItemBoolFn = bool(*)(const lang::ModPathView& path);
		ItemBoolFn itemExistsPtr;
	};
	struct CompCfg
	{
		//you need to add a newline yourself, if you need to (use println or something)
		//Note: msg may contain nulls
		using LogFn = void(*)(const std::string_view msg);
		LogFn logPtr;

		using GetFileContentsFn = std::optional<std::vector<uint8_t>>(*)(const std::string_view path);
		GetFileContentsFn getFileContentsPtr;

		using FileBoolFn = bool(*)(const std::string_view path);
		FileBoolFn fileExistsPtr;
		FileBoolFn isFolderPtr;
		FileBoolFn isSymLinkPtr;

		//file paths inside a path, return empty list if path is invalid or doesnt exist
		using GetFileListFn = std::vector<std::string>(*)(const std::string_view path);
		GetFileListFn getFileListPtr;
		GetFileListFn getFileListRecPtr;//recursive version

		std::span<const CompPlugin> plugins;

		std::span<const std::string_view> rootPaths;//All of them must have some crate.

		//will use exactly that many threads, not any less, not any more
		size_t extraThreadCount = 
			(size_t)std::max(//max, so no negative thread counts
				int64_t(std::thread::hardware_concurrency()) 
					- 3,//2 sys threads, 1 for main thread
				0LL
		);
	};
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
	};

	struct SluFile
	{
		std::string_view crateRootPath;
		std::string path;
		std::vector<uint8_t> contents;
	};

	namespace CompTaskType
	{
		using ParseFiles = std::vector<SluFile>;
		struct ConsensusMergeAsts
		{
			std::vector<std::string> allPaths;
		};
	}
	using CompTaskData = std::variant<
		CompTaskType::ParseFiles,
		CompTaskType::ConsensusMergeAsts
	>;
	struct CompTask
	{
		CompTaskData data;
		size_t threadsLeft : 31 = 0;
		size_t taskId : 32 = 0;
		size_t leaveForMain : 1 = false;
	};

	inline void poolThread(const CompCfg& cfg,
		std::atomic_bool& shouldExit,
		std::condition_variable& cv,
		Mutex<std::condition_variable>& cvMain,
		std::atomic_size_t& tasksLeft,
		Mutex<std::vector<CompTask>>& tasks
	)
	{
		uint32_t lastTask = 0;
		while(true)
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

			if (isCompleterThread && !taskRef.leaveForMain)
			{
				stackCopy = std::move(taskRef.data);
				tasks.v.pop_back();
				taskLock.unlock();//only thread with this stuff, so no need to keep a lock!
			}
			CompTaskData& task = stackCopy.has_value()
				? stackCopy.value() 
				: taskRef.data;//else, it was not invalidated

			//Complete it...



			if (isCompleterThread)
			{
				if (tasksLeft.fetch_sub(1, std::memory_order_relaxed) - 1 == 0)
				{
					//wake up, all tasks are done!
					cvMain.v.notify_all();
				}
			}
		}
	}

	inline CompOutput compile(const CompCfg& cfg)
	{
		uint32_t nextTaskId = 1;
		Mutex<std::vector<CompTask>> tasks;
		std::atomic_size_t tasksLeft;

		std::condition_variable cv;
		Mutex<std::condition_variable> cvMain;
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
				size_t filesPerThread = sluFiles.size() / cfg.extraThreadCount;
				if (filesPerThread == 0) filesPerThread = 1;

				size_t total = sluFiles.size();
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

					tasksLeft++;
					{
						std::unique_lock _(tasks.lock);
						tasks.v.emplace_back(std::move(res));
					}
					cv.notify_one();
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
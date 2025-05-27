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

namespace slu::comp
{

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
		std::condition_variable& cvMain,
		Mutex<size_t>& tasksLeft,
		Mutex<std::vector<CompTask>>& tasks
	)
	{
		uint32_t lastTask = 0;
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
				std::lock_guard _(tasksLeft.lock);
				if (--tasksLeft.v == 0)
				{
					//wake up, all tasks are done!
					cvMain.notify_all();
				}
			}
		}
	}
}
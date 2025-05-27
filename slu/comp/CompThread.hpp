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
		TaskHandleState state;
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
				if (--tasksLeft.v == 0)
				{
					//wake up, all tasks are done!
					cvMain.notify_all();
				}
			}
		}
	}
}
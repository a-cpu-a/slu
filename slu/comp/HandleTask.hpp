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
#include <slu/ext/CppMatch.hpp>

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
	using InputType = slu::parse::VecInput<decltype(slu::parse::sluCommon)>;
	struct ParsedFile
	{
		slu::parse::ParsedFile<InputType> parsed;
		std::string_view crateRootPath;
		std::string path;
	};
	struct TaskHandleState
	{
		std::vector<ParsedFile> parsedFiles;
	};

	inline void handleTask(const CompCfg& cfg,
		std::atomic_bool& shouldExit,
		TaskHandleState& state,
		CompTaskData& task
	)
	{
		ezmatch(task)(
		varcase(CompTaskType::ParseFiles&) {
			// Handle parsing files
			for (SluFile& file : var) 
			{//cfg, file.crateRootPath, file.path, file.contents
				InputType in;
				in.fName = file.path;
				in.text = file.contents;

				ParsedFile parsed;
				parsed.parsed = slu::parse::parseFile(in);
				parsed.crateRootPath = file.crateRootPath;
				parsed.path = std::move(file.path);

				state.parsedFiles.emplace_back(std::move(parsed));
			}
		},
		varcase(CompTaskType::ConsensusMergeAsts&) {
			// Handle consensus merging of ASTs
			for (const auto& path : var.allPaths)
			{
				//TODO: Implement actual merging logic
			}
		}
		);
	}
}
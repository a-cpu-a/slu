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
#include <slu/midlevel/BasicDesugar.hpp>

#include <slu/comp/CompCfg.hpp>
#include <slu/comp/lua/Conv.hpp>

namespace slu::comp
{
	using SettingsType = decltype(slu::parse::sluCommon);
	using InputType = slu::parse::VecInput<SettingsType>;

	struct SluFile
	{
		std::string_view crateRootPath;
		std::string path;
		std::vector<uint8_t> contents;
	};
	using GenCodeMap = std::unordered_map<uint32_t, std::vector<std::vector<uint8_t>>>;
	namespace CompTaskType
	{
		using ParseFiles = std::vector<SluFile>;
		struct ConsensusUnifyAsts
		{
			RwLock<parse::BasicMpDbData>* sharedDb;//stored on main thread.
			bool firstToArive = true; //if true, then you dont need to do any logic, just replace it
		};
		struct ConsensusMergeAsts
		{
			std::unordered_map<std::string, int> path2Ast;
		};
		struct DoCodeGen
		{
			std::vector<std::span<const parse::Statement<InputType>>> statements;
			uint32_t entrypointId;
		};
		using ConsensusMergeGenCode = GenCodeMap*;
	}
	using CompTaskData = std::variant<
		CompTaskType::ParseFiles,
		CompTaskType::ConsensusUnifyAsts,
		CompTaskType::ConsensusMergeAsts,
		CompTaskType::DoCodeGen,
		CompTaskType::ConsensusMergeGenCode
	>;
	struct CompTask
	{
		CompTaskData data;
		size_t threadsLeft : 32 = 0;
		size_t taskId : 32 = 0;
	};
	struct ParsedFile
	{
		slu::parse::ParsedFile<InputType> parsed;
		std::string_view crateRootPath;
		std::string path;
	};
	struct TaskHandleState
	{
		const parse::BasicMpDbData* sharedDb=nullptr; // Set after ConsensusUnifyAsts
		std::vector<ParsedFile> parsedFiles;
		parse::BasicMpDbData mpDb;
		std::unordered_map<uint32_t, std::vector<uint8_t>> genOut;
	};

	inline lang::ModPath parsePath(std::string_view crateRootPath, std::string_view path)
	{
		lang::ModPath res;

		for (size_t i = crateRootPath.size(); i > 0; i--)
		{
			bool slash = crateRootPath[i - 1] == '/';
			if (slash || i==1)
			{
				res.push_back(std::string(crateRootPath.substr(i 
					+ (slash?0:-1)// if it is not a slash, then also include it too
				)));
				break;
			}
		}

		size_t pathStart = crateRootPath.size()+1;//+1 for slash
		bool skipFolder = true;// skip "src", etc
		for (size_t i = 0; i < path.size(); i++)
		{
			if (i < crateRootPath.size())
			{
				if (path[i] != crateRootPath[i])
					break;//TODO: error, maybe log it
			}
			else if (i == crateRootPath.size())
			{
				if (path[i] != '/')
					break;//TODO: error, maybe log it
			}
			else
			{
				if (path[i] == '/')
				{
					if (pathStart == i)// found a '//' ?
						break;//TODO: error, maybe log it
					if(!skipFolder)
						res.push_back(std::string(path.substr(pathStart, i-pathStart)));
					skipFolder = false;
				}
				pathStart++;
			}
		}
		return res;
	}

	inline void handleTask(const CompCfg& cfg,
		std::atomic_bool& shouldExit,
		TaskHandleState& state,
		std::unique_lock<std::mutex>* taskLock,//if null, then already unlocked
		CompTaskData& task
	)
	{
		ezmatch(task)(
		varcase(CompTaskType::ParseFiles&) 
		{ // Handle parsing files
			state.parsedFiles.reserve(var.size());
			for (SluFile& file : var) 
			{//cfg, file.crateRootPath, file.path, file.contents
				InputType in;
				in.fName = file.path;
				in.text = file.contents;
				in.genData.mpDb = { &state.mpDb };
				in.genData.totalMp = parsePath(file.crateRootPath,file.path);

				ParsedFile parsed;
				parsed.parsed = slu::parse::parseFile(in);
				slu::mlvl::basicDesugar(state.mpDb,parsed.parsed);
				parsed.crateRootPath = file.crateRootPath;
				parsed.path = std::move(file.path);

				state.parsedFiles.emplace_back(std::move(parsed));
			}
		},
		varcase(CompTaskType::ConsensusUnifyAsts&) 
		{ // Handle consensus unification of ASTs
			auto& sharedDb = *var.sharedDb;
			state.sharedDb = &sharedDb.v;

			if (taskLock != nullptr)
			{
				if (var.firstToArive)
				{//Easy
					var.sharedDb->v = std::move(state.mpDb);
					var.firstToArive = false;
					return;
				}
				taskLock->unlock();
				// var is gone now!!!
			}
			throw std::runtime_error("TODO: unify state.mpDb!");
		},
		varcase(CompTaskType::ConsensusMergeAsts&) 
		{ // Handle consensus merging of ASTs
			for (const auto& path : var.path2Ast)
			{
				//TODO: Implement actual merging logic
			}
		},
		varcase(CompTaskType::DoCodeGen&) 
		{ // Handle code gen of all the global statements
			auto& outVec = state.genOut[var.entrypointId];
			parse::Output out;
			parse::LuaMpDb luaDb;
			out.text = std::move(outVec);
			for (const auto& i : var.statements)
			{
				for (const auto& j : i)
				{
					slu::comp::lua::conv({
						CommonConvData{cfg,*state.sharedDb,j},
						luaDb, out 
					});
				}
			}
			outVec = std::move(out.text);
		},
		varcase(CompTaskType::ConsensusMergeGenCode&) {
			for (auto& [epId, i] : state.genOut)
			{
				(*var)[epId].emplace_back(std::move(i));
			}
		}
		);
	}
}
/*
    Slu language compiler, a computer program compiler.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;

//#include <slu/comp/mlir/Pch.hpp>
export module slu.comp.handle_task;

import a_cpu_a.mtx;
import slu.settings;
import slu.ast.state;
import slu.ast.mp_data;
import slu.comp.cfg;
import slu.comp.conv_data;
import slu.comp.mlir.conv;
import slu.lang.basic_state;
import slu.mlvl.basic_desugar;
import slu.mlvl.type_inf_check;
import slu.parse.error;
import slu.parse.parse;
import slu.parse.vec_input;

namespace slu::comp
{
	using SettingsType = decltype(slu::parse::sluCommon);
	using InputType = slu::parse::VecInput<SettingsType>;

	export struct SluFile
	{
		std::string_view crateRootPath;
		std::string path;
		std::vector<uint8_t> contents;
	};
	export struct ParsedFile
	{
		lang::ModPath mp; //Only valid before MergeAstsMap!!!
		std::string_view crateRootPath;
		std::string path;
		parse::ParsedFile pf;
	};
	export using GenCodeMap
	    = std::unordered_map<uint32_t, std::vector<llvm::SmallVector<char, 0>>>;
	export using MergeAstsMap = std::unordered_map<lang::ModPath, ParsedFile,
	    lang::HashModPathView, lang::EqualModPathView>;
	export struct FileStatList
	{
		std::span<const parse::Stat> stats;
		std::string_view filePath;
	};
	export struct MutFileStatList
	{
		std::span<parse::Stat> stats;
		std::string_view filePath;
	};
	namespace CompTaskType
	{
		export using ParseFiles = std::vector<SluFile>;
		export struct ConsensusUnifyAsts
		{
			a_cpu_a::RwLock<parse::BasicMpDbData>*
			    sharedDb;             //stored on main thread.
			bool firstToArive = true; //if true, then you dont need to do any
			                          //logic, just replace it
		};
		export using ConsensusMergeAsts = MergeAstsMap*;
		export struct DoCodeGen
		{
			std::vector<FileStatList> statements;
			uint32_t entrypointId;
		};
		export using TypeInfCheck = std::vector<MutFileStatList>;
		export using ConsensusMergeGenCode = GenCodeMap*;
	} //namespace CompTaskType
	export using CompTaskData = std::variant<CompTaskType::ParseFiles,
	    CompTaskType::ConsensusUnifyAsts, CompTaskType::ConsensusMergeAsts,
	    CompTaskType::DoCodeGen, CompTaskType::TypeInfCheck,
	    CompTaskType::ConsensusMergeGenCode>;
	export struct CompTask
	{
		CompTaskData data;
		size_t threadsLeft : 32 = 0;
		size_t taskId      : 32 = 0;
	};
	export struct TaskHandleStateMlir
	{
		mlir::MLIRContext mc;
		llvm::LLVMContext llvmCtx;
		mlir::OpBuilder opBuilder{&mc};
		mlir::PassManager pm{&mc};
		//mlir::FrozenRewritePatternSet patterns;
		mlir::DataLayoutEntryAttr indexLay;
	};
	export struct TaskHandleState
	{
		TaskHandleStateMlir* s;

		const parse::BasicMpDbData* sharedDb
		    = nullptr; // Set after ConsensusUnifyAsts
		std::vector<ParsedFile> parsedFiles;
		parse::BasicMpDbData mpDb;
		std::unordered_map<uint32_t, std::unique_ptr<llvm::Module>> genOut;

		mlir::LLVMConversionTarget target;
		mlir::LLVMTypeConverter tyConv;

		std::vector<mico::MpElementInfo> mp2Elements;

		uint32_t indexSize = 64;
	};

	inline lang::ModPath parsePath(
	    std::string_view crateRootPath, std::string_view path)
	{
		lang::ModPath res;

		for (size_t i = crateRootPath.size(); i > 0; i--)
		{
			bool slash = crateRootPath[i - 1] == '/';
			if (slash || i == 1)
			{
				res.push_back(std::string(crateRootPath.substr(
				    i + (slash ? 0 : -1) // if it is not a slash, then also
				                         // include it too
				    )));
				break;
			}
		}

		size_t pathStart = crateRootPath.size() + 1; //+1 for slash
		bool skipFolder = true;                      // skip "src", etc
		for (size_t i = 0; i < path.size(); i++)
		{
			if (i < crateRootPath.size())
			{
				if (path[i] != crateRootPath[i])
					break; //TODO: error, maybe log it
			} else if (i == crateRootPath.size())
			{
				if (path[i] != '/')
					break; //TODO: error, maybe log it
			} else
			{
				if (path[i] == '/')
				{
					if (pathStart == i) // found a '//' ?
						break;          //TODO: error, maybe log it
					if (!skipFolder)
						res.push_back(
						    std::string(path.substr(pathStart, i - pathStart)));
					skipFolder = false;
				}
				pathStart++;
			}
		}
		return res;
	}

	export void handleTask(const CompCfg& cfg, std::atomic_bool& shouldExit,
	    TaskHandleState& state,
	    std::unique_lock<std::mutex>* taskLock, //if null, then already unlocked
	    CompTaskData& task)
	{
		ezmatch(task)(
		    varcase(CompTaskType::ParseFiles&) { // Handle parsing files
			    state.parsedFiles.reserve(var.size());
			    for (SluFile& file : var)
			    { //cfg, file.crateRootPath, file.path, file.contents
				    InputType in;
				    in.fName = file.path;
				    in.text = file.contents;
				    in.genData.mpDb = {&state.mpDb};
				    in.genData.totalMp
				        = parsePath(file.crateRootPath, file.path);

				    ParsedFile parsed;
				    try
				    {
					    parsed.pf = slu::parse::parseFile(in);
				    } catch (const slu::parse::ParseFailError&)
				    {
					    cfg.logPtr("Failed to parse: " + file.path);
					    for (auto& i : in.handledErrors)
					    {
						    cfg.logPtr(i);
						    i.clear(); // Free it faster
					    }
				    }
				    std::vector<mlvl::InlineModule> inlineModules
				        = slu::mlvl::basicDesugar(state.mpDb, parsed.pf);

				    parsed.crateRootPath = file.crateRootPath;
				    parsed.path = file.path;
				    parsed.mp = parsePath(file.crateRootPath, file.path);
				    state.parsedFiles.emplace_back(std::move(parsed));

				    for (mlvl::InlineModule& i : inlineModules)
				    {
					    parsed.pf.code = std::move(i.code);
					    parsed.crateRootPath = file.crateRootPath;
					    parsed.path = file.path;
					    parsed.mp = state.sharedDb->getMp(i.name);
					    // ^ Will be correct, as name is the name of the local
					    // obj inside of the root module.
					    //a::b, b is the local obj (also the modules name), and
					    //its still included in getMp's result.
					    state.parsedFiles.emplace_back(std::move(parsed));
				    }
			    }
		    },
		    varcase(CompTaskType::ConsensusUnifyAsts&) { // Handle consensus
			                                             // unification of ASTs
			    auto& sharedDb = *var.sharedDb;
			    state.sharedDb = &sharedDb.v;

			    if (var.firstToArive)
			    { //Easy
				    var.sharedDb->v = std::move(state.mpDb);
				    var.firstToArive = false;
				    return;
			    }
			    if (taskLock != nullptr)
				    taskLock->unlock();
			    // var is gone now!!!

			    throw std::runtime_error("TODO: unify state.mpDb!");
		    },
		    varcase(CompTaskType::ConsensusMergeAsts) { // Handle consensus
			                                            // merging of ASTs
			    for (auto& i : state.parsedFiles)
			    {
				    var->emplace(std::move(i.mp),
				        std::move(i) //mp was moved, so it is now invalid.
				    );
			    }
			    state.parsedFiles.clear(); // Not needed anymore
		    },
		    varcase(CompTaskType::TypeInfCheck&) {
			    for (const auto& i : var)
				    mlvl::typeInferAndCheck(*state.sharedDb, i.stats);
		    },
		    varcase(CompTaskType::DoCodeGen&) { // Handle code gen of all the
			                                    // global statements
			    //parse::Output out;
			    //parse::LuaMpDb luaDb;
			    //out.text = std::move(outVec);

			    auto module = mlir::ModuleOp::create(
			        state.s->opBuilder.getUnknownLoc(), "HelloWorldModule");

			    mlir::OpBuilder& builder = state.s->opBuilder;

			    module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
			        mlir::DataLayoutSpecAttr::get(
			            &state.s->mc, {state.s->indexLay}));
			    builder.setInsertionPointToStart(module.getBody());

			    using namespace std::literals::string_view_literals;
			    auto privVis = builder.getStringAttr("private"sv);
			    //auto publVis = builder.getStringAttr("nested"sv);
			    //auto nestVis = builder.getStringAttr("nested"sv);

			    auto data = mico::ConvData{
			        CommonConvData{cfg, *state.sharedDb},
                    state.mp2Elements,
			        state.s->mc, state.s->llvmCtx, state.s->opBuilder,
			        state.tyConv, module, privVis
                };

			    for (const auto& i : var.statements)
			    {
				    data.filePath = i.filePath;
				    for (const auto& j : i.stats)
				    {
					    /*slu::comp::lua::conv({
					        CommonConvData{cfg,*state.sharedDb,j},
					        luaDb, out
					    });*/
					    data.stat = &j;
					    slu::comp::mico::conv(data);
					    state.mp2Elements.clear(); //The module doesnt exist
					                               //anymore, cant reuse them.
				    }
			    }
			    module.print(llvm::outs());

			    //if (mlir::failed(mlir::applyFullConversion(module,
			    //state.target, state.s->patterns)))
			    //{
			    //	cfg.errPtr("Failed to apply full conversion for entrypoint:
			    //" + std::to_string(var.entrypointId));
			    //	//todo: filename!
			    //	return;
			    //}
			    if (mlir::failed(state.s->pm.run(module)))
			    {
				    cfg.errPtr("Failed to run pass manager for entrypoint: "
				        + std::to_string(var.entrypointId));
				    //todo: filename!
				    return;
			    }
			    cfg.logPtr("=== CONVERTED ===");
			    module.print(llvm::outs());

			    auto llvmMod = mlir::translateModuleToLLVMIR(
			        module, state.s->llvmCtx, "HelloWorldLlvmModule");
			    if (!llvmMod)
			    {
				    cfg.errPtr(
				        "Failed to translate module to Llvm Ir for entrypoint: "
				        + std::to_string(var.entrypointId));
				    //todo: filename!
				    return;
			    }
			    llvmMod->print(llvm::outs(), nullptr);

			    auto p = state.genOut.find(var.entrypointId);
			    if (p != state.genOut.end())
			    {
				    llvm::Linker linker(*p->second);
				    if (linker.linkInModule(std::move(llvmMod)))
				    {
					    cfg.errPtr("Failed to link module for entrypoint: "
					        + std::to_string(var.entrypointId));
					    return;
				    }
				    cfg.logPtr("==== LINK ====");
				    p->second->print(llvm::outs(), nullptr);
			    } else
				    state.genOut[var.entrypointId] = std::move(llvmMod);
		    },
		    varcase(CompTaskType::ConsensusMergeGenCode&) {
			    for (auto& [epId, module] : state.genOut)
			    {
				    //module to .o file
				    std::string error;
				    llvm::Triple targetTriple{
				        llvm::sys::getDefaultTargetTriple()};
				    const llvm::Target* target
				        = llvm::TargetRegistry::lookupTarget(
				            targetTriple, error);
				    if (!target)
				    {
					    cfg.errPtr("Failed to find target: " + error);
					    continue;
				    }

				    llvm::TargetOptions opt;
				    auto rm = std::optional<llvm::Reloc::Model>();
				    std::unique_ptr<llvm::TargetMachine> targetMachine(
				        target->createTargetMachine(
				            targetTriple, "generic", "", opt, rm));

				    module->setDataLayout(targetMachine->createDataLayout());
				    module->setTargetTriple(targetTriple);

				    llvm::SmallVector<char, 0> objBuffer;
				    llvm::raw_svector_ostream objStream(objBuffer);

				    llvm::legacy::PassManager pass;
				    if (targetMachine->addPassesToEmitFile(pass, objStream,
				            nullptr, llvm::CodeGenFileType::ObjectFile))
				    {
					    throw std::runtime_error(
					        "TargetMachine can't emit a file of this type");
				    }

				    pass.run(*module);

				    (*var)[epId].emplace_back(std::move(objBuffer));
			    }
		    });
	}
} //namespace slu::comp
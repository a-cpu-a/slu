/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <thread>
#include <variant>
#include <slu/ext/CppMatch.hpp>
// Mlir / llvm includes
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4146)
#pragma warning(disable : 4267)

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/CSE.h>
#include <mlir/Transforms/Passes.h>
#include <llvm/InitializePasses.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <lld/Common/Driver.h>

LLD_HAS_DRIVER(elf);
LLD_HAS_DRIVER(wasm);
LLD_HAS_DRIVER(coff);

#pragma warning(pop)

#include <slu/lang/BasicState.hpp>

#include <slu/comp/CompCfg.hpp>
#include <slu/comp/ConvData.hpp>

namespace slu::comp::mico
{
	using namespace std::string_view_literals;

	struct LocalStackItm
	{
		size_t itemCount;
		std::vector<mlir::Value> values;
	};

	struct GlobalElement
	{
		//TODO: mlir::Value, type, function, etc
		mlir::func::FuncOp func;
		std::string_view abi;
	};
	//localObj2CachedData
	using MpElementInfo = std::unordered_map<size_t, GlobalElement>;

	struct ConvData : CommonConvData
	{
		std::vector<MpElementInfo>& mp2Elements;
		mlir::MLIRContext& context;
		llvm::LLVMContext& llvmContext;
		mlir::OpBuilder& builder;
		mlir::ModuleOp module;
		mlir::StringAttr privVis;
		std::vector<LocalStackItm> localsStack;
		uint64_t nextPrivTmpId = 0;

		void addLocalStackItem(const size_t itmCount) {
			localsStack.emplace_back(itmCount);
			localsStack.back().values.resize(itmCount);
		}
		GlobalElement* getElement(const lang::MpItmIdV<true> name)
		{
			if(name.mp.id>= mp2Elements.size())
				return nullptr;//not found
			auto& mp = mp2Elements[name.mp.id];
			auto it = mp.find(name.id.val);
			if (it == mp.end())
				return nullptr;//not found
			return &it->second;
		}
		GlobalElement* addElement(const lang::MpItmIdV<true> name, mlir::func::FuncOp func, std::string_view abi)
		{
			if (name.mp.id >= mp2Elements.size())
				mp2Elements.resize(name.mp.id + 1);
			auto& mp = mp2Elements[name.mp.id];
			mp[name.id.val] = {.func= func,.abi=abi};
			return &mp[name.id.val];
		}
	};
	//Forward declare!
	mlir::Value convExpr(ConvData& conv, const parse::ExpressionV<true>& itm, const std::string_view abi = ""sv);
	//

	mlir::StringAttr getExportAttr(ConvData& conv,const bool exported) {
		return exported ? mlir::StringAttr() : conv.privVis;
	}
	mlir::Location convPos(ConvData& conv,parse::Position p)
	{
		return mlir::FileLineColLoc::get(
			&conv.context,
			"someFile.slu",
			(uint32_t)p.line,
			uint32_t(p.index + 1) // mlir is 1-based, not 0-based
		);
	}

	struct TmpName
	{
		std::array<char, 3 + 8> store;

		constexpr std::string_view sref() const {
			return std::string_view{ store.data(), store.size() };
		}
		constexpr operator std::string_view() const {
			return sref();
		}
	};
	// _C_XXXXXXXX
	constexpr TmpName mkTmpName(ConvData& conv)
	{
		TmpName res;
		res.store[0] = '_';
		res.store[1] = 'C';
		res.store[2] = '_';
		uint64_t v = conv.nextPrivTmpId++;
		for (size_t i = 0; i < 8; i++)
		{
			res.store[3 + i] = parse::numToHex(v & 0xF);
			v >>= 4;
		}
		return res;
	}
	struct ViewOrStr
	{
		std::variant<std::string, std::string_view> val;
		constexpr std::string_view sv()const {
			return std::visit([](const auto& v) { return std::string_view{ v }; }, val);
		}
	};
	ViewOrStr mangleFuncName(ConvData& conv, const std::string_view abi,lang::MpItmIdV<true> name)
	{
		if(abi=="C")
			return { name.asSv(conv.sharedDb) };
		auto vmp = name.asVmp(conv.sharedDb);

		if (vmp.back() == "main")
			return { "main"sv };

		//Construct a mangled name.
		std::string res;
		size_t totalChCount = 0;
		for (auto& i : vmp)
			totalChCount += i.size();
		res.reserve(2+2* vmp.size()+totalChCount); // :>::aaa::bbb::ccc

		res.push_back(':');
		res.push_back('>');
		for (auto& i : vmp)
		{
			res.push_back(':');
			res.push_back(':');
			res.append(i);
		}
		return { std::move(res)};
	}

	mlir::Type tryConvBuiltinType(ConvData& conv, const std::string_view abi, const parse::LimPrefixExprV<true>& itm,const bool reffed)
	{
		const auto& var = std::get<parse::LimPrefixExprType::VARv<true>>(itm).v;
		if(!var.sub.empty())
			throw std::runtime_error("Unimplemented type expression (has subvar's) (mlir conversion)");
		const auto& name = std::get<parse::BaseVarType::NAMEv<true>>(var.base).v;
		
		if (name == conv.sharedDb.getItm({ "std","str" }))
		{
			if(!reffed)
				throw std::runtime_error("Unimplemented type expression: std::str, without ref (mlir conversion, reffed expected)");

			if(abi=="C")
				return mlir::LLVM::LLVMPointerType::get(&conv.context);

			auto i8Type = conv.builder.getIntegerType(8);
			return mlir::MemRefType::get({ -1 }, i8Type, {}, 0);
		}
		if (name == conv.sharedDb.getItm({ "std","i32" }))
		{
			auto i32Type = conv.builder.getIntegerType(32);
			if (reffed)
				return mlir::MemRefType::get({ 1 }, i32Type, {}, 0);
			return i32Type;
		}
		if (name == conv.sharedDb.getItm({ "std","void" }))
			return mlir::NoneType::get(&conv.context);

		throw std::runtime_error("Unimplemented type expression: " + std::string(name.asSv(conv.sharedDb)) + " (mlir conversion)");
	}
	mlir::Type convTypeHack(ConvData& conv,const std::string_view abi,const parse::ExpressionV<true>& expr)
	{
		return ezmatch(expr.data)(
		varcase(const auto&)->mlir::Type
		{
			throw std::runtime_error("Unimplemented type expression idx(" + std::to_string(expr.data.index()) + ") (mlir conversion)");
		},
		varcase(const parse::ExprType::LimPrefixExprV<true>&)
		{
			return tryConvBuiltinType(conv, abi, *var, false);
		},
		varcase(const parse::ExprType::FuncCallV<true>&)
		{
			if(var.argChain.size()!=1)
				throw std::runtime_error("Unimplemented type expression: function call with multiple layers (mlir conversion)");
			
			const auto& func = std::get<parse::LimPrefixExprType::VARv<true>>(*var.val).v;
			if (!func.sub.empty())
				throw std::runtime_error("Unimplemented type expression (has subvar's) (mlir conversion)");
			const auto& name = std::get<parse::BaseVarType::NAMEv<true>>(func.base).v;
			if(name!= conv.sharedDb.getItm({ "std","ops","Ref","ref"}))
				throw std::runtime_error("Unimplemented type expression: " + std::string(name.asSv(conv.sharedDb)) + " (mlir conversion)");
			
			const auto& expArgs = std::get<parse::ArgsType::EXPLISTv<true>>(var.argChain[0].args).v;
			const auto& firstArgExpr = expArgs.front();
			const auto& firstArg = std::get<parse::ExprType::LimPrefixExprV<true>>(firstArgExpr.data);
			
			return tryConvBuiltinType(conv, abi, *firstArg, true);
		}
		);
	}
	template<bool forStore>
	inline mlir::Value convVarBase(ConvData& conv, parse::Position place, const parse::BaseVarV<true>& itm, const std::string_view abi)
	{
		auto* mc = &conv.context;
		mlir::OpBuilder& builder = conv.builder;

		return ezmatch(itm)(
		varcase(const parse::BaseVarType::Local) {
			mlir::Value alloc = conv.localsStack.back().values[var.v];

			if constexpr (forStore)return alloc;

			const mlir::Location loc = convPos(conv, place);
			mlir::Value index0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
			mlir::Value loaded = builder.create<mlir::memref::LoadOp>(loc, alloc, mlir::ValueRange{ index0 });

			if (abi == "C"sv && loaded.getType().isIndex())
			{
				auto i64Type = builder.getIntegerType(64);
				auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(mc);

				// %ptrInt = index.castu %ptrIdx : index to i64
				auto ptrInt = builder.create<mlir::index::CastUOp>(
					convPos(conv, place), i64Type, loaded
				);
				// %llvm_ptr = llvm.inttoptr %ptrInt : i64 to !llvm.ptr
				return (mlir::Value)builder.create<mlir::LLVM::IntToPtrOp>(convPos(conv, place),
					llvmPtrType, ptrInt
					, nullptr
					//, mlir::LLVM::DereferenceableAttr::get(mc, str.size(), false)
				);
			}

			return loaded;
		},
		varcase(const parse::BaseVarType::NAMEv<true>&) {
			//TODO
			return (mlir::Value)nullptr;
		},
		varcase(const parse::BaseVarType::EXPRv<true>&) {
			return convExpr(conv,var.start);
		},
		varcase(const parse::BaseVarType::Root) {
			//TODO: builtin root-mp reflection value
			auto i1Type = builder.getIntegerType(1);
			return (mlir::Value)builder.create<mlir::arith::ConstantOp>(
				convPos(conv, place), i1Type, mlir::IntegerAttr::get(i1Type, 0)
			);
		}
		);
	}
	inline mlir::Value convLimPrefixExpr(ConvData& conv,parse::Position place, const parse::LimPrefixExprV<true>& itm, const std::string_view abi)
	{
		return ezmatch(itm)(
		varcase(const parse::LimPrefixExprType::EXPRv<true>&){
			return convExpr(conv, var.v,abi);
		},
		varcase(const parse::LimPrefixExprType::VARv<true>&){
			return convVarBase<false>(conv, place, var.v.base,abi);
				//TODO sub;
			//basicly (.x == memref subview?) (.y.x().x == multiple stuff)
		}
		);
	}

	template<typename T>
	concept AnyInvalidExpression =
		std::same_as<T, parse::ExprType::TRUE>
		|| std::same_as<T, parse::ExprType::FALSE>
		|| std::same_as<T, parse::ExprType::NIL>
		|| std::same_as<T, parse::ExprType::MULTI_OPERATIONv<true>>;//already desugared
	inline mlir::Value convAny64(ConvData& conv, parse::Position place, const parse::Any64BitInt auto itm)
	{
		mlir::OpBuilder& builder = conv.builder;
		auto i64Type = builder.getIntegerType(64);
		return builder.create<mlir::arith::ConstantOp>(
			convPos(conv, place), i64Type, mlir::IntegerAttr::get(i64Type, (int64_t)itm)
		);
	}
	inline mlir::Value convAny128(ConvData& conv, parse::Position place, const parse::Any128BitInt auto itm)
	{
		mlir::OpBuilder& builder = conv.builder;
		auto i128Type = builder.getIntegerType(128);
		llvm::APInt apVal(128, llvm::ArrayRef{ itm.lo ,itm.hi });
		return builder.create<mlir::arith::ConstantOp>(
			convPos(conv, place), i128Type, mlir::IntegerAttr::get(i128Type, apVal)
		);
	}
	inline mlir::Value convExpr(ConvData& conv, const parse::ExpressionV<true>& itm, const std::string_view abi)
	{
		auto* mc = &conv.context;
		mlir::OpBuilder& builder = conv.builder;


		return ezmatch(itm.data)(

		varcase(const auto&)->mlir::Value {
			conv.cfg.errPtr(std::to_string(itm.data.index()));
			throw std::runtime_error("Unimplemented expression type idx(" + std::to_string(itm.data.index()) + ") (mlir conversion)");
		},
		varcase(const parse::ExprType::I64) {return convAny64(conv,itm.place,var); },
		varcase(const parse::ExprType::U64) {return convAny64(conv,itm.place,var); },
		varcase(const parse::ExprType::I128) {return convAny128(conv,itm.place,var); },
		varcase(const parse::ExprType::U128) {return convAny128(conv,itm.place,var); },

		varcase(const parse::ExprType::LimPrefixExprV<true>&)->mlir::Value {
			return convLimPrefixExpr(conv,itm.place,*var,abi);
		},
		varcase(const parse::ExprType::String&)->mlir::Value {
		
			auto i8Type = builder.getIntegerType(8);
			auto i64Type = builder.getIntegerType(64);
			auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(mc);
			auto strType = mlir::MemRefType::get({ (int64_t)var.v.size() }, i8Type, {}, 0);
		
			TmpName strName = mkTmpName(conv);
			{
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.setInsertionPointToStart(conv.module.getBody());
		
				auto denseStr = mlir::DenseElementsAttr::get(strType, llvm::ArrayRef{ var.v.data(),var.v.size() });
				builder.create<mlir::memref::GlobalOp>(
					convPos(conv, itm.place),
					strName.sref(),
					/*sym_visibility=*/conv.privVis,
					strType,
					denseStr,
					/*constant=*/true,
					/*alignment=*/builder.getIntegerAttr(i8Type, 1)
				);
			}
		
			// %str = memref.get_global @greeting : memref<14xi8>
			auto globalStr = builder.create<mlir::memref::GetGlobalOp>(convPos(conv, itm.place), strType,
				strName.sref());
			// %ptrIdx = memref.extract_aligned_pointer_as_index %str
			auto ptrIndex = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
				convPos(conv, itm.place), globalStr
			);

			if (abi != "C"sv)
				return ptrIndex;
		
			// %ptrInt = index.castu %ptrIdx : index to i64
			auto ptrInt = builder.create<mlir::index::CastUOp>(
				convPos(conv, itm.place), i64Type, ptrIndex
			);
			// %llvm_ptr = llvm.inttoptr %ptrInt : i64 to !llvm.ptr
			return builder.create<mlir::LLVM::IntToPtrOp>(convPos(conv, itm.place),
				llvmPtrType, ptrInt
				, nullptr
				//, mlir::LLVM::DereferenceableAttr::get(mc, str.size(), false)
			);
		},


			//Ignore these
		varcase(const AnyInvalidExpression auto&)->mlir::Value {
			throw std::runtime_error("Invalid expression type idx(" + std::to_string(itm.data.index()) + ") (mlir conversion)");
		}
		);
	}

	template<typename T>
	concept AnyIgnoredStatement =
		std::same_as<T, parse::StatementType::GOTOv<true>>
		|| std::same_as<T, parse::StatementType::SEMICOLON>
		|| std::same_as<T, parse::StatementType::USE>
		|| std::same_as<T, parse::StatementType::FnDeclV<true>>
		|| std::same_as<T, parse::StatementType::FunctionDeclV<true>>
		|| std::same_as<T, parse::StatementType::ExternBlockV<true>>//ignore, as desugaring will remove it
		|| std::same_as<T, parse::StatementType::UnsafeBlockV<true>>//ignore, as desugaring will remove it
		|| std::same_as<T, parse::StatementType::DROPv<true>>
		|| std::same_as<T, parse::StatementType::MOD_DEFv<true>>
		|| std::same_as<T, parse::StatementType::MOD_DEF_INLINEv<true>>
		|| std::same_as<T, parse::StatementType::UNSAFE_LABEL>
		|| std::same_as<T, parse::StatementType::SAFE_LABEL>;

	inline void convSoeOrBlock(ConvData& conv, const parse::SoeOrBlockV<true>& itm)
	{

	}
	inline void convStat(ConvData& conv, const parse::StatementV<true>& itm)
	{
		auto* mc = &conv.context;
		mlir::OpBuilder& builder = conv.builder;


		ezmatch(itm.data)(

			varcase(const auto&) {
			throw std::runtime_error("Unimplemented statement type idx(" + std::to_string(itm.data.index()) + ") (mlir conversion)");
		},
			varcase(const parse::StatementType::CanonicLocal&) {
			mlir::Value val = convExpr(conv, var.value);

			const mlir::Location loc = convPos(conv, itm.place);
			auto memrefType = mlir::MemRefType::get({ 1 }, val.getType()); // memref<1x_>
			mlir::Value alloc = builder.create<mlir::memref::AllocaOp>(loc, memrefType);


			mlir::Value index0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
			builder.create<mlir::memref::StoreOp>(loc, val, alloc, mlir::ValueRange{ index0 });

			conv.localsStack.back().values[var.name.v] = alloc;
		},
			varcase(const parse::StatementType::REPEAT_UNTILv<true>&) {
			mlir::Type i1Type = builder.getI1Type();
			const mlir::Location loc = convPos(conv, itm.place);
			// Initial loop-carried value: run once → %keepGoing = true
			mlir::Value one = builder.create<mlir::arith::ConstantIntOp>(loc, 1, i1Type);

			auto whileOp = builder.create<mlir::scf::WhileOp>(loc, mlir::TypeRange{ i1Type }, mlir::ValueRange{ one });

			auto condArg = whileOp.getBefore().addArgument(i1Type, loc);
			builder.setInsertionPointToStart(whileOp.getBeforeBody());
			builder.create<mlir::scf::ConditionOp>(loc, condArg, mlir::ValueRange{ condArg });

			whileOp.getAfter().addArgument(i1Type, loc);
			builder.setInsertionPointToStart(whileOp.getAfterBody());
			for (const auto& i : var.bl.statList)
				convStat(conv, i);
			if (var.bl.hadReturn && !var.bl.retExprs.empty())
			{
				//maybe return something
				//TODO
			}

			// repeat-until: stop if cond is true → so loop if NOT result
			mlir::Value continueLoop = builder.create<mlir::arith::XOrIOp>(
				loc, convExpr(conv,var.cond), one);

			builder.create<mlir::scf::YieldOp>(convPos(conv, var.bl.end), mlir::ValueRange{ continueLoop });
		},
			varcase(const parse::StatementType::WHILE_LOOPv<true>&) {
			auto whileOp = builder.create<mlir::scf::WhileOp>(convPos(conv,itm.place), mlir::TypeRange{}, mlir::ValueRange{});

			builder.setInsertionPointToStart(whileOp.getBeforeBody());
			builder.create<mlir::scf::ConditionOp>(convPos(conv, var.cond.place), convExpr(conv,var.cond), mlir::ValueRange{});

			builder.setInsertionPointToStart(whileOp.getAfterBody());
			for (const auto& i : var.bl.statList)
				convStat(conv, i);
			if (var.bl.hadReturn && !var.bl.retExprs.empty())
			{
				//maybe return something
				//TODO
			}
			builder.create<mlir::scf::YieldOp>(convPos(conv,var.bl.end));
		},
			varcase(const parse::StatementType::IfCondV<true>&) {
			
			std::vector<mlir::Block*> yieldBlocks;
			yieldBlocks.reserve(var.elseIfs.size() + (var.elseBlock.has_value() ? 1 : 0));
			size_t elIfIdx = 0;
			mlir::scf::IfOp firstOp;
			do {
				const parse::ExpressionV<true>* cond;
				const parse::SoeOrBlockV<true>* bl;
				mlir::Location loc = nullptr;
				bool hasMore = false;

				if (elIfIdx == 0)
				{
					cond = &*var.cond;
					bl = &var.bl.get();
					loc = convPos(conv, itm.place);
					hasMore = var.elseBlock.has_value() || !var.elseIfs.empty();
				}
				else
				{
					const auto& elIf = var.elseIfs[elIfIdx - 1];
					cond = &elIf.first;
					bl = &elIf.second;
					loc = convPos(conv, elIf.first.place);
					hasMore = var.elseBlock.has_value() || (var.elseIfs.size() > elIfIdx);
				}

				auto op = builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange{},
					convExpr(conv,*cond), hasMore);

				if (elIfIdx == 0)
					firstOp = op;

				builder.setInsertionPointToStart(op.thenBlock());
				convSoeOrBlock(conv, *bl);
				builder.create<mlir::scf::YieldOp>(convPos(conv, itm.place));
				if(hasMore)
				{
					builder.setInsertionPointToStart(op.elseBlock());
					yieldBlocks.push_back(op.elseBlock());
				}
			} while (elIfIdx++ < var.elseIfs.size());

			if (var.elseBlock.has_value())
				convSoeOrBlock(conv, var.elseBlock.value().get());

			for (const auto i : yieldBlocks)
			{
				builder.setInsertionPointToStart(i);
				builder.create<mlir::scf::YieldOp>(convPos(conv, itm.place));
			}

			builder.setInsertionPointAfter(firstOp);
		},
			varcase(const parse::StatementType::ASSIGNv<true>&) {
			if (var.vars.size() != 1 || var.exprs.size() != 1)
				throw std::runtime_error("Unimplemented assign conv, vers.size or var.exprs != 1");

			if(!var.vars[0].sub.empty())
				throw std::runtime_error("Unimplemented assign conv, var sub !empty");

			mlir::Value memRef = convVarBase<true>(conv, itm.place,
				var.vars[0].base,
				""sv
			);
			const mlir::Location loc = convPos(conv, itm.place);
			auto zeroIndex = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);

			builder.create<mlir::memref::StoreOp>(loc, convExpr(conv, var.exprs[0], ""sv), memRef, mlir::ValueRange{ zeroIndex }, false);

		},
			varcase(const parse::StatementType::FuncCallV<true>&) {

			auto& varInfo = std::get<parse::LimPrefixExprType::VARv<true>>(*var.val).v.base;
			auto name = std::get<parse::BaseVarType::NAMEv<true>>(varInfo);
			GlobalElement* funcInfo = conv.getElement(name.v);

			auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(mc);
			//It cant be the other types, as they are alreadt desugared!
			auto& argList = std::get<parse::ArgsType::EXPLISTv<true>>(var.argChain[0].args).v;
			std::vector<mlir::Value> args;
			args.reserve(argList.size());
			for (const auto& i : argList)
				args.emplace_back(convExpr(conv, i,funcInfo->abi));
			
			// %res = llvm.call @puts(%llvm_ptr) : (!llvm.ptr) -> i32
			builder.create<mlir::LLVM::CallOp>(
				convPos(conv, itm.place),
				funcInfo->func.getResultTypes(),
				mlir::SymbolRefAttr::get(mc, funcInfo->func.getSymName()),
				llvm::ArrayRef<mlir::Value>{args});

		},
		varcase(const parse::StatementType::CONSTv<true>&) {

			conv.addLocalStackItem(var.local2Mp.size());

			const auto str = "Hello world\0"sv;
			auto i8Type = builder.getIntegerType(8);
			auto strType = mlir::MemRefType::get({ (int64_t)str.size() }, i8Type, {}, 0);
			
			auto denseStr = mlir::DenseElementsAttr::get(strType, llvm::ArrayRef{ str.data(),str.size() });

			builder.create<mlir::memref::GlobalOp>(
				convPos(conv, itm.place),
				llvm::StringRef{ ":>::hello_world::greeting"sv },
				/*sym_visibility=*/getExportAttr(conv, var.exported),
				strType,
				denseStr,
				/*constant=*/true,
				/*alignment=*/builder.getIntegerAttr(i8Type, 1)
			);

			conv.localsStack.pop_back();

		},
		varcase(const parse::StatementType::FnDeclV<true>&) {

			mlir::OpBuilder::InsertionGuard guard(builder);
			builder.setInsertionPointToStart(conv.module.getBody());

			auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(mc);
			//TODO: use type converter!
			auto putsType = builder.getFunctionType({ llvmPtrType }, { convTypeHack(conv, var.abi, *var.retType.value()) });
			//StringRef  name, FunctionType type, ArrayRef<NamedAttribute> attrs = {}, ArrayRef<DictionaryAttr> argAttrs = {});
			//StringRef  sym_name, ::mlir::FunctionType function_type, /*optional*/::mlir::StringAttr sym_visibility, /*optional*/::mlir::ArrayAttr arg_attrs, /*optional*/::mlir::ArrayAttr res_attrs, /*optional*/bool no_inline = false);

			auto mangledName = mangleFuncName(conv, var.abi, var.name); 
			mlir::func::FuncOp decl = builder.create<mlir::func::FuncOp>(convPos(conv, itm.place),
				mangledName.sv(),
				putsType, getExportAttr(conv, var.exported),
				nullptr, nullptr, false
			);
			conv.addElement(var.name, decl, var.abi);
		},
		varcase(const parse::StatementType::FNv<true>&) {

			mlir::OpBuilder::InsertionGuard guard(builder);
			builder.setInsertionPointToStart(conv.module.getBody());

			GlobalElement* funcInfo = conv.getElement(var.name);
			if (funcInfo == nullptr)
			{
				auto mangledName = mangleFuncName(conv, var.func.abi, var.name);

				mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(
					convPos(conv, var.place), mangledName.sv(),
					mlir::FunctionType::get(mc, {}, {}),
					getExportAttr(conv, var.exported),
					nullptr, nullptr, false
				);
				funcInfo = conv.addElement(var.name, funcOp, var.func.abi);
			}

			// Build a function in mlir
			conv.addLocalStackItem(var.func.local2Mp.size());


			mlir::Block* entry = funcInfo->func.addEntryBlock();
			builder.setInsertionPointToStart(entry);

			for (auto& i : var.func.block.statList)
				convStat(conv, i);

			if (var.func.block.hadReturn && !var.func.block.retExprs.empty())
			{
				//maybe return something
				//TODO
				
			}
			
			builder.create<mlir::func::ReturnOp>(convPos(conv, var.func.block.end));

			conv.localsStack.pop_back();
		},


			//Ignore these
			varcase(const AnyIgnoredStatement auto&) {}
			);
	}
	inline void conv(ConvData& conv) {
		convStat(conv, *conv.stat);
	}
}
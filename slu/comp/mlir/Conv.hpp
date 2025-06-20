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

	struct ConvData : CommonConvData
	{
		mlir::MLIRContext& context;
		llvm::LLVMContext& llvmContext;
		mlir::OpBuilder& builder;
		mlir::ModuleOp module;
		uint64_t nextPrivTmpId = 0;
	};

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


	template<typename T>
	concept AnyInvalidExpression =
		std::same_as<T, parse::ExprType::TRUE>
		|| std::same_as<T, parse::ExprType::FALSE>
		|| std::same_as<T, parse::ExprType::NIL>
		|| std::same_as<T, parse::ExprType::MULTI_OPERATIONv<true>>;//already desugared

	inline mlir::Value convExpr(ConvData& conv, const parse::ExpressionV<true>& itm)
	{
		auto* mc = &conv.context;
		mlir::OpBuilder& builder = conv.builder;


		return ezmatch(itm.data)(

		varcase(const auto&)->mlir::Value {
			throw std::runtime_error("Unimplemented expression type idx(" + std::to_string(itm.data.index()) + ") (mlir conversion)");
		},
			//TODO: are these ignored, or does it actually work?
		varcase(const auto)->mlir::Value requires (parse::Any64BitInt<decltype(var)>) {
			auto i64Type = builder.getIntegerType(64);
			return builder.create<mlir::arith::ConstantOp>(
				builder.getUnknownLoc(), i64Type, mlir::IntegerAttr::get(i64Type, (int64_t)var.v)
			);
		},
		varcase(const auto)->mlir::Value requires (parse::Any128BitInt<decltype(var)>) {
			auto i128Type = builder.getIntegerType(128);
			llvm::APInt apVal(128, llvm::ArrayRef{ var.lo ,var.hi });
			return builder.create<mlir::arith::ConstantOp>(
				builder.getUnknownLoc(), i128Type, mlir::IntegerAttr::get(i128Type, apVal)
			);
			return {};
		},
		varcase(const parse::ExprType::LIM_PREFIX_EXPv<true>&)->mlir::Value {
			//TODO: implement this
			auto i1Type = builder.getIntegerType(1);
			return builder.create<mlir::arith::ConstantOp>(
				builder.getUnknownLoc(), i1Type, mlir::IntegerAttr::get(i1Type, 0)
			);
		},
		varcase(const parse::ExprType::LITERAL_STRING&)->mlir::Value {
		
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
					builder.getUnknownLoc(),
					strName.sref(),
					/*sym_visibility=*/builder.getStringAttr("private"sv),
					strType,
					denseStr,
					/*constant=*/true,
					/*alignment=*/builder.getIntegerAttr(i8Type, 1)
				);
			}
		
			// %str = memref.get_global @greeting : memref<14xi8>
			auto globalStr = builder.create<mlir::memref::GetGlobalOp>(builder.getUnknownLoc(), strType,
				strName.sref());
			// %ptrIdx = memref.extract_aligned_pointer_as_index %str
			auto ptrIndex = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
				builder.getUnknownLoc(), globalStr
			);
		
			// %ptrInt = index.castu %ptrIdx : index to i64
			auto ptrInt = builder.create<mlir::index::CastUOp>(
				builder.getUnknownLoc(), i64Type, ptrIndex
			);
			// %llvm_ptr = llvm.inttoptr %ptrInt : i64 to !llvm.ptr
			return builder.create<mlir::LLVM::IntToPtrOp>(builder.getUnknownLoc(),
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

	inline void convStat(ConvData& conv, const parse::StatementV<true>& itm)
	{
		auto* mc = &conv.context;
		mlir::OpBuilder& builder = conv.builder;

		const auto str = "Hello world\0"sv;
		auto i8Type = builder.getIntegerType(8);
		auto strType = mlir::MemRefType::get({ (int64_t)str.size() }, i8Type, {}, 0);

		ezmatch(itm.data)(

			varcase(const auto&) {
			throw std::runtime_error("Unimplemented statement type idx(" + std::to_string(itm.data.index()) + ") (mlir conversion)");
		},
			varcase(const parse::StatementType::FUNC_CALLv<true>&) {

			auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(mc);
#if 0
			auto i64Type = builder.getIntegerType(64);

			// %str = memref.get_global @greeting : memref<14xi8>
			auto globalStr = builder.create<mlir::memref::GetGlobalOp>(builder.getUnknownLoc(), strType,
				":>::hello_world::greeting"sv);


			// mlir::Type aligned_pointer, ::mlir::Value source);
			// mlir::Value source);
			// mlir::TypeRange resultTypes, ::mlir::Value source);
			// mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
			// mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
			// mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties & properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
			// mlir::ValueRange operands, const Properties & properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
			// 
			// %ptrIdx = memref.extract_aligned_pointer_as_index %str
			auto ptrIndex = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
				builder.getUnknownLoc(), globalStr
			);

			// %ptrInt = index.castu %ptrIdx : index to i64
			auto ptrInt = builder.create<mlir::index::CastUOp>(
				builder.getUnknownLoc(), i64Type, ptrIndex
			);

			//// %ptrInt = arith.index_cast %ptrIdx : index to i64
			//auto ptrInt = builder.create<mlir::arith::IndexCastOp>(
			//	builder.getUnknownLoc(), i64Type, ptrIndex
			//);

			// mlir::Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
			// mlir::Type res, ::mlir::Value arg, /*optional*/::mlir::LLVM::DereferenceableAttr dereferenceable);
			// mlir::TypeRange resultTypes, ::mlir::Value arg, /*optional*/::mlir::LLVM::DereferenceableAttr dereferenceable);
			// mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
			// mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties & properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
			// 
			// %llvm_ptr = llvm.inttoptr %ptrInt : i64 to !llvm.ptr
			auto llvmPtr = builder.create<mlir::LLVM::IntToPtrOp>(builder.getUnknownLoc(),
				llvmPtrType, ptrInt
				, nullptr
				//, mlir::LLVM::DereferenceableAttr::get(mc, str.size(), false)
			);
			#endif

			//It cant be the other types, as they are alreadt desugared!
			auto& argList = std::get<parse::ArgsType::EXPLISTv<true>>(var.argChain[0].args).v;
			std::vector<mlir::Value> args;
			args.reserve(argList.size());
			for (const auto& i : argList)
				args.emplace_back(convExpr(conv, i));

			// %res = llvm.call @puts(%llvm_ptr) : (!llvm.ptr) -> i32
			builder.create<mlir::LLVM::CallOp>(
				builder.getUnknownLoc(),
				builder.getI32Type(),
				mlir::SymbolRefAttr::get(mc, "puts"sv),
				llvm::ArrayRef<mlir::Value>{args});

		},
			varcase(const parse::StatementType::CONSTv<true>&) {

			//auto strAttr = builder.getStringAttr(llvm::Twine{ std::string_view{str,strLen} });
			auto denseStr = mlir::DenseElementsAttr::get(strType, llvm::ArrayRef{ str.data(),str.size() });

			builder.create<mlir::memref::GlobalOp>(
				builder.getUnknownLoc(),
				llvm::StringRef{ ":>::hello_world::greeting"sv },
				/*sym_visibility=*/builder.getStringAttr("private"sv),
				strType,
				denseStr,
				/*constant=*/true,
				/*alignment=*/builder.getIntegerAttr(i8Type, 1)
			);

		},
			varcase(const parse::StatementType::FnDeclV<true>&) {
			auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(mc);
			auto putsType = builder.getFunctionType({ llvmPtrType }, { builder.getI32Type() });
			builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "puts"sv, putsType).setPrivate();

		},
			varcase(const parse::StatementType::FNv<true>&) {
			// Build a function in mlir

			auto loc = mlir::FileLineColLoc::get(mc, builder.getStringAttr("myfile.sv"sv), (uint32_t)var.place.line, (uint32_t)var.place.index);

			mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(
				loc, var.name.asSv(conv.sharedDb),
				mlir::FunctionType::get(mc, {}, {})
			);
			mlir::Block* entry = funcOp.addEntryBlock();
			builder.setInsertionPointToStart(entry);

			for (auto& i : var.func.block.statList)
				convStat(conv, i);

			builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

		},


			//Ignore these
			varcase(const AnyIgnoredStatement auto&) {}
			);
	}
	inline void conv(ConvData&& conv) {
		convStat(conv, conv.stat);
	}
}
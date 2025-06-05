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
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Support/LogicalResult.h>
#include <llvm/InitializePasses.h>
#include <llvm/IR/LLVMContext.h>

#include <slu/lang/BasicState.hpp>

#include <slu/comp/CompCfg.hpp>
#include <slu/comp/ConvData.hpp>

namespace slu::comp::mico
{
	template<typename T>
	concept AnyIgnoredStatement = 
		std::same_as<T,parse::StatementType::GOTOv<true>>
		|| std::same_as<T,parse::StatementType::SEMICOLON>
		|| std::same_as<T,parse::StatementType::USE>
		|| std::same_as<T,parse::StatementType::FnDeclV<true>>
		|| std::same_as<T,parse::StatementType::FunctionDeclV<true>>
		|| std::same_as<T,parse::StatementType::ExternBlockV<true>>//ignore, as desugaring will remove it
		|| std::same_as<T,parse::StatementType::UnsafeBlockV<true>>//ignore, as desugaring will remove it
		|| std::same_as<T,parse::StatementType::DROPv<true>>
		|| std::same_as<T,parse::StatementType::MOD_DEFv<true>>
		|| std::same_as<T,parse::StatementType::MOD_DEF_INLINEv<true>>
		|| std::same_as<T,parse::StatementType::UNSAFE_LABEL>
		|| std::same_as<T,parse::StatementType::SAFE_LABEL>;

	struct ConvData : CommonConvData
	{
		mlir::MLIRContext& context;
		llvm::LLVMContext& llvmContext;
	};

	inline void convStat(const ConvData& conv)
	{
		auto* mc = &conv.context;
		ezmatch(conv.stat.data)(

		varcase(const auto&) {},
		varcase(const parse::StatementType::FNv<true>&) {
			// Build a function in mlir

			mlir::OpBuilder builder(mc);

			const char str[] = "Hello world";
			int64_t strLen = sizeof(str);
			auto strType = mlir::MemRefType::get({ strLen }, builder.getIntegerType(8), {}, 0);
			auto strAttr = builder.getStringAttr(llvm::Twine{ std::string_view{str,strLen} });


			builder.create<mlir::LLVM::GlobalOp>(
				builder.getUnknownLoc(),
				strType,
				/*isConstant=*/true,
				mlir::LLVM::Linkage::Internal,
				"hello_str",
				strAttr
			);

			auto loc = mlir::FileLineColLoc::get(mc,builder.getStringAttr("myfile.sv"), var.place.line, var.place.index);

			mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(
				loc, var.name.asSv(conv.sharedDb),
				mlir::FunctionType::get(mc, {}, {})
			);
		},


			//Ignore these
		varcase(const AnyIgnoredStatement auto&) {}
		);
	}
	inline void conv(const ConvData& conv)
	{
		//TODO: Implement the conversion logic here
	}
}
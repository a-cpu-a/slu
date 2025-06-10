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
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Support/LogicalResult.h>
#include <llvm/InitializePasses.h>
#include <llvm/IR/LLVMContext.h>
#pragma warning(pop)

#include <slu/lang/BasicState.hpp>

#include <slu/comp/CompCfg.hpp>
#include <slu/comp/ConvData.hpp>

namespace slu::comp::mico
{
	using namespace std::string_view_literals;

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
		mlir::OpBuilder& builder;
	};

	inline void convStat(const ConvData& conv,const parse::StatementV<true>& itm)
	{
		auto* mc = &conv.context;
		ezmatch(itm.data)(

		varcase(const auto&) {},
		varcase(const parse::StatementType::FNv<true>&) {
			// Build a function in mlir

			mlir::OpBuilder& builder = conv.builder;

			auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
			builder.setInsertionPointToStart(module.getBody());

			const auto str = "Hello world\0"sv;
			auto i8Type = builder.getIntegerType(8);
			auto strType = mlir::MemRefType::get({ (int64_t)str.size() }, builder.getIntegerType(8), {}, 0);
			//auto strAttr = builder.getStringAttr(llvm::Twine{ std::string_view{str,strLen} });
			auto denseStr = mlir::DenseElementsAttr::get(strType, llvm::ArrayRef{ str.data(),str.size()});


			// ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::TypeAttr type, /*optional*/::mlir::Attribute initial_value, /*optional*/::mlir::UnitAttr constant, /*optional*/::mlir::IntegerAttr alignment);
			// ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::TypeAttr type, /*optional*/::mlir::Attribute initial_value, /*optional*/::mlir::UnitAttr constant, /*optional*/::mlir::IntegerAttr alignment);
			// ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::MemRefType type, /*optional*/::mlir::Attribute initial_value, /*optional*/bool constant, /*optional*/::mlir::IntegerAttr alignment);
			// ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::MemRefType type, /*optional*/::mlir::Attribute initial_value, /*optional*/bool constant, /*optional*/::mlir::IntegerAttr alignment);

			builder.create<mlir::memref::GlobalOp>(
				builder.getUnknownLoc(),
				"greeting"sv,
				/*sym_visibility=*/builder.getStringAttr("private"sv),
				strType,
				denseStr,
				/*constant=*/true
			);


			auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(mc);
			auto putsType = builder.getFunctionType({ llvmPtrType }, { builder.getI32Type() });
			builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "puts"sv, putsType).setPrivate();

			auto loc = mlir::FileLineColLoc::get(mc,builder.getStringAttr("myfile.sv"sv), (uint32_t)var.place.line, (uint32_t)var.place.index);

			mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(
				loc, var.name.asSv(conv.sharedDb),
				mlir::FunctionType::get(mc, {}, {})
			);
			mlir::Block* entry = funcOp.addEntryBlock();
			builder.setInsertionPointToStart(entry);

			// %str = memref.get_global @greeting : memref<14xi8>
			auto globalStr = builder.create<mlir::memref::GetGlobalOp>(builder.getUnknownLoc(), strType, "greeting"sv);

			// %ptr = memref.extract_aligned_pointer_as_index %str
			auto ptrIndex = builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
				builder.getUnknownLoc(), builder.getIndexType(), globalStr
			);


			// %llvm_ptr = llvm.inttoptr %ptr : index to !llvm.ptr<i8>
			auto llvmPtr = builder.create<mlir::LLVM::IntToPtrOp>(builder.getUnknownLoc(), llvmPtrType, ptrIndex);


			// %res = llvm.call @puts(%llvm_ptr) : (!llvm.ptr<i8>) -> i32
			builder.create<mlir::LLVM::CallOp>(
				builder.getUnknownLoc(),
				builder.getI32Type(),
				mlir::SymbolRefAttr::get(mc, "puts"sv),
				llvm::ArrayRef<mlir::Value>{llvmPtr});

			builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

			module.print(llvm::outs());
		},


			//Ignore these
		varcase(const AnyIgnoredStatement auto&) {}
		);
	}
	inline void conv(const ConvData& conv) {
		convStat(conv,conv.stat);
	}
}
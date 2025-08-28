module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <string>
#include <vector>
#include <optional>
#include <thread>
#include <variant>

#include <slu/comp/mlir/IncludeMlir.hpp>
#include <slu/ext/CppMatch.hpp>
export module slu.comp.mlir.conv;

import slu.char_info;
import slu.ast.mp_data;
import slu.ast.pos;
import slu.ast.state_decls;
import slu.ast.state;
import slu.ast.type;
import slu.comp.cfg;
import slu.comp.conv_data;
import slu.lang.basic_state;
import slu.lang.mpc;

LLD_HAS_DRIVER(elf);
LLD_HAS_DRIVER(wasm);
LLD_HAS_DRIVER(coff);

namespace slu::comp::mico
{
	using namespace std::string_view_literals;

	export struct LocalStackVal
	{
		mlir::Value v;
		const parse::ResolvedType* ty;
	};
	export struct LocalStackItm
	{
		size_t itemCount;
		std::vector<LocalStackVal> values;
	};

	namespace GlobElemTy
	{
		export struct Fn
		{
			mlir::func::FuncOp func;
			std::string_view abi;
		};
		export struct Alias {
			lang::MpItmId realName;
		};
		//TODO: mlir::Value, type, etc
	}

	export using GlobElem = std::variant<GlobElemTy::Fn, GlobElemTy::Alias>;
	//localObj2CachedData
	export using MpElementInfo = std::unordered_map<size_t, GlobElem>;

	export struct ConvData : CommonConvData
	{
		std::vector<MpElementInfo>& mp2Elements;
		mlir::MLIRContext& context;
		llvm::LLVMContext& llvmContext;
		mlir::OpBuilder& builder;
		mlir::LLVMTypeConverter& tyConv;
		mlir::ModuleOp module;
		mlir::StringAttr privVis;
		std::vector<LocalStackItm> localsStack;
		uint64_t nextPrivTmpId = 0;

		void addLocalStackItem(const size_t itmCount) {
			localsStack.emplace_back(itmCount);
			localsStack.back().values.resize(itmCount);
		}
		GlobElem* getElement(const lang::MpItmId name)
		{
			if(name.mp.id>= mp2Elements.size())
				return nullptr;//not found
			auto& mp = mp2Elements[name.mp.id];
			auto it = mp.find(name.id.val);
			if (it == mp.end())
				return nullptr;//not found
			return &it->second;
		}
		GlobElemTy::Fn* addElem(const lang::MpItmId name, mlir::func::FuncOp func, std::string_view abi)
		{
			if (name.mp.id >= mp2Elements.size())
				mp2Elements.resize(name.mp.id + 1);
			auto& mp = mp2Elements[name.mp.id];
			mp[name.id.val] = GlobElemTy::Fn{.func= func,.abi=abi};
			return &std::get<GlobElemTy::Fn>(mp[name.id.val]);
		}
		GlobElemTy::Alias* addElem(const lang::MpItmId name, const lang::MpItmId realName)
		{
			if (name.mp.id >= mp2Elements.size())
				mp2Elements.resize(name.mp.id + 1);
			auto& mp = mp2Elements[name.mp.id];
			mp[name.id.val] = GlobElemTy::Alias{.realName = realName };
			return &std::get<GlobElemTy::Alias>(mp[name.id.val]);
		}
	};
	//Forward declare!
	mlir::Value convExpr(ConvData& conv, ast::Position place, const parse::ExprDataV<true>& itm);
	void convStat(ConvData& conv, const parse::Stat& itm);
	//

	mlir::StringAttr getExportAttr(ConvData& conv,const bool exported) {
		return exported ? mlir::StringAttr() : conv.privVis;
	}
	mlir::Location convPos(ConvData& conv,ast::Position p)
	{
		return mlir::FileLineColLoc::get(
			&conv.context,
			conv.filePath,
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
			res.store[3 + i] = slu::numToHex(v & 0xF);
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
	ViewOrStr mangleFuncName(ConvData& conv, const std::string_view abi,lang::MpItmId name)
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

	mlir::Type tryConvBuiltinType(ConvData& conv, const std::string_view abi, const lang::MpItmId& name,const bool reffed)
	{
		if (name == mpc::STD_STR)
		{
			if(!reffed)
				throw std::runtime_error("Unimplemented type expression: std::str, without ref (mlir conversion, reffed expected)");

			if(abi=="C")
				return mlir::LLVM::LLVMPointerType::get(&conv.context);

			auto i8Type = conv.builder.getI8Type();
			return mlir::MemRefType::get({ mlir::ShapedType::kDynamic }, i8Type, {}, 0);
		}
		if (name == mpc::STD_I32)
		{
			auto i32Type = conv.builder.getI32Type();
			if (reffed)
				return mlir::MemRefType::get({ 1 }, i32Type, {}, 0);
			return i32Type;
		}
		if (name == mpc::STD_UNIT)
			return mlir::NoneType::get(&conv.context);

		throw std::runtime_error("Unimplemented type expression: " + std::string(name.asSv(conv.sharedDb)) + " (mlir conversion)");
	}
	mlir::Type convTypeHack(ConvData& conv,const std::string_view abi,const parse::Expr& expr)
	{
		return ezmatch(expr.data)(
		varcase(const auto&)->mlir::Type
		{
			throw std::runtime_error("Unimplemented type expression idx(" + std::to_string(expr.data.index()) + ") (mlir conversion)");
		},
		varcase(const parse::ExprType::GlobalV<true>&)
		{
			return tryConvBuiltinType(conv, abi, var, false);
		},
		varcase(const parse::ExprType::SelfCall&)
		{
			if (var.method != conv.sharedDb.getItm({ "std","ops","Ref","ref" }))
				throw std::runtime_error("Unimplemented type expression: " + std::string(var.method.asSv(conv.sharedDb)) + " (mlir conversion)");

			const auto& selfName = std::get<parse::ExprType::GlobalV<true>>(var.v->data);

			return tryConvBuiltinType(conv, abi, selfName, true);
		},
		varcase(const parse::ExprType::Call&)->mlir::Type
		{
			auto name = std::get<parse::ExprType::GlobalV<true>>(var.v->data);
			throw std::runtime_error("Unimplemented type expression: " + std::string(name.asSv(conv.sharedDb)) + " (mlir conversion)");
			
			//const auto& expArgs = std::get<parse::ArgsType::ExprList>(var.args);
			//const auto& firstArgExpr = expArgs.front();
			//const auto& firstArg = std::get<parse::ExprType::GlobalV<true>>(firstArgExpr.data);
			//
			//return tryConvBuiltinType(conv, abi, firstArg, true);
		}
		);
	}

	mlir::Type convType(ConvData& conv, const parse::ResolvedType& itm, const std::string_view abi)
	{
		if(!itm.isComplete())
			throw std::runtime_error("Found incomplete type (mlir conversion)");

		mlir::OpBuilder& builder = conv.builder;

		mlir::Type elemType;
		const bool cAbi = abi == "C"sv;
		const bool elemUnsized = itm.size == parse::ResolvedType::UNSIZED_MARK;
		if(elemUnsized)
		{
			if(cAbi)
				throw std::runtime_error("Found unsized type in C ABI (mlir conversion)");
			elemType = builder.getI8Type();
		}
		else
		{
			if (cAbi && itm.size == parse::TYPE_RES_PTR_SIZE && itm.outerSliceDims == 0 && std::holds_alternative<parse::RawTypeKind::RefChain>(itm.base))
			{//Treat references & pointers as llvm.ptr.
				elemType = mlir::LLVM::LLVMPointerType::get(&conv.context);
			}
			else
				elemType = builder.getIntegerType(itm.size);
		}

		if (cAbi)
		{
			if (itm.outerSliceDims != 0)
				throw std::runtime_error("Found unsized slice type in C ABI (mlir conversion)");
			return elemType;
		}

		//Memref {?x}elemType
		llvm::SmallVector<int64_t> shape;
		if (itm.outerSliceDims == 0)
			shape.push_back(elemUnsized? mlir::ShapedType::kDynamic :1);
		else
		{
			shape.append(
				itm.outerSliceDims + (elemUnsized ? 1 : 0),
				mlir::ShapedType::kDynamic
			);
		}
		return mlir::MemRefType::get(shape, elemType, {}, 0);
	}
	template<typename T>
	concept AnyInvalidExpression =
		std::same_as<T, parse::ExprType::True>
		|| std::same_as<T, parse::ExprType::False>
		|| std::same_as<T, parse::ExprType::Nil>
		|| std::same_as<T, parse::ExprType::MultiOpV<true>>;//already desugared
	inline mlir::Value convAny64(ConvData& conv, ast::Position place, const parse::Any64BitInt auto itm)
	{
		mlir::OpBuilder& builder = conv.builder;
		auto i64Type = builder.getI64Type();
		return mlir::arith::ConstantOp::create(builder,
			convPos(conv, place), i64Type, mlir::IntegerAttr::get(i64Type, (int64_t)itm)
		);
	}
	inline mlir::Value convAny128(ConvData& conv, ast::Position place, const parse::Any128BitInt auto itm)
	{
		mlir::OpBuilder& builder = conv.builder;
		auto i128Type = builder.getIntegerType(128);
		llvm::APInt apVal(128, llvm::ArrayRef{ itm.lo ,itm.hi });
		return mlir::arith::ConstantOp::create(builder,
			convPos(conv, place), i128Type, mlir::IntegerAttr::get(i128Type, apVal)
		);
	}
	inline mlir::Value convExpr(ConvData& conv,ast::Position place, const parse::ExprDataV<true>& itm)
	{
		auto* mc = &conv.context;
		mlir::OpBuilder& builder = conv.builder;

		return ezmatch(itm)(

		varcase(const auto&)->mlir::Value {
			conv.cfg.errPtr(std::to_string(itm.index()));
			throw std::runtime_error("Unimplemented expression type idx(" + std::to_string(itm.index()) + ") (mlir conversion)");
		},
		varcase(const parse::ExprType::I64) {return convAny64(conv,place,var); },
		varcase(const parse::ExprType::U64) {return convAny64(conv,place,var); },
		varcase(const parse::ExprType::P128) {return convAny128(conv,place,var); },
		varcase(const parse::ExprType::M128) {return convAny128(conv,place,var); },

		varcase(const parse::ExprType::ParensV<true>&) {
			return convExpr(conv,var->place,var->data);
		},
		varcase(const parse::ExprType::Local) {
			return conv.localsStack.back().values[var.v].v;
		},
		varcase(const parse::ExprType::FieldV<true>&) {

			mlir::Value base = convExpr(conv, var.v->place, var.v->data);

			if (var.field == conv.sharedDb.getPoolStr("__convHack__refSlice_ptr"sv))
			{
				//Hack: take out ptr.

				mlir::Type idx3Type = mlir::MemRefType::get({ 3 }, builder.getIndexType(), {}, 0);
				mlir::Type rawMemref = conv.tyConv.convertType(idx3Type);

				mlir::Location loc = convPos(conv, place);


				mlir::Value lForm = mlir::UnrealizedConversionCastOp::create(builder,
					loc, mlir::TypeRange{ rawMemref }, mlir::ValueRange{ base }
				).getResult(0);
				mlir::Value idx3Form = mlir::UnrealizedConversionCastOp::create(builder,
					loc, mlir::TypeRange{ idx3Type }, mlir::ValueRange{ lForm }
				).getResult(0);

				mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
				mlir::Value idxPtr = mlir::memref::LoadOp::create(builder, loc, idx3Form, mlir::ValueRange{ c0 }, false, 0ULL);

				mlir::Value intPtr = mlir::index::CastUOp::create(builder, loc, builder.getI64Type(), idxPtr);

				return mlir::LLVM::IntToPtrOp::create(builder, loc, mlir::LLVM::LLVMPointerType::get(&conv.context), intPtr).getRes();
			}
			//TODO sub;
			//basicly (.x == memref subview?) (.y.x().x == multiple stuff)
			throw std::runtime_error("TODO: Field expr's are unimplemented (mlir conversion)");
		},
		varcase(const parse::ExprType::String&)->mlir::Value {
		
			auto i8Type = builder.getI8Type();
			auto i64Type = builder.getI64Type();
			auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(mc);
			auto strType = mlir::MemRefType::get({ (int64_t)var.v.size() }, i8Type, {}, 0);

			constexpr uint32_t REF_SLICE_SIZE = parse::TYPE_RES_PTR_SIZE + parse::TYPE_RES_SIZE_SIZE * 2;
			auto ptrNsizeX2Int = builder.getIntegerType(REF_SLICE_SIZE);
			auto refSliceBytesType = mlir::MemRefType::get({ (REF_SLICE_SIZE + 7) / 8 }, i8Type, {}, 0);
			auto refSliceType = mlir::MemRefType::get({ 1 }, ptrNsizeX2Int, {}, 0);
			auto idx3Type = mlir::MemRefType::get({ 3 }, builder.getIndexType(), {}, 0);

			auto tensorType = mlir::RankedTensorType::get({ (int64_t)var.v.size() }, i8Type);
			auto denseStr = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef{ (const int8_t*)var.v.data(),var.v.size() });

			mlir::Location loc = convPos(conv, place);

			TmpName strName = mkTmpName(conv);
			{
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.setInsertionPointToStart(conv.module.getBody());
		
				mlir::memref::GlobalOp::create(builder,
					loc,
					strName.sref(),
					/*sym_visibility=*/conv.privVis,
					strType,
					denseStr,
					/*constant=*/true,
					/*alignment=*/builder.getIntegerAttr(i64Type, 1)
				);
			}
		
			// %str = memref.get_global @strName : memref<14xi8>
			auto globalStr = mlir::memref::GetGlobalOp::create(builder,loc, strType,
				strName.sref());

			mlir::Value refSliceAlloc = mlir::memref::AllocaOp::create(builder,
				loc, refSliceType
			);

			//Reinterpret as 3xindex.
			mlir::Value cv0 = mlir::arith::ConstantIntOp::create(builder,loc, i64Type,0);
			mlir::Value cv3 = mlir::arith::ConstantIntOp::create(builder,loc, i64Type,3);
			mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder,loc, 0);
			mlir::Value c1 = mlir::arith::ConstantIndexOp::create(builder,loc, 1);
			mlir::Value c2 = mlir::arith::ConstantIndexOp::create(builder,loc, 2);
			mlir::Value c3 = mlir::arith::ConstantIndexOp::create(builder,loc, 3);


			mlir::Value ptr = mlir::memref::ExtractAlignedPointerAsIndexOp::create(builder,loc, globalStr);
			auto data = mlir::memref::ExtractStridedMetadataOp::create(builder,loc, globalStr);

			//mlir::Value idx3Form = slu_dial::ReinterpretMemRefOp::create(builder,
			//	loc,idx3Type,refSliceAlloc
			//);
			mlir::Value idx3Form = mlir::UnrealizedConversionCastOp::create(builder,
				loc, mlir::TypeRange{ idx3Type }, mlir::ValueRange{ refSliceAlloc }
			).getResult(0);


			// TODO: check if offset matters. NOTE: wont matter in this specific use case.
			
			mlir::memref::StoreOp::create(builder,loc, ptr, idx3Form, mlir::ValueRange{c0});
			mlir::memref::StoreOp::create(builder,loc, data.getSizes()[0], idx3Form, mlir::ValueRange{c1});
			mlir::memref::StoreOp::create(builder,loc, data.getStrides()[0], idx3Form, mlir::ValueRange{c2});

			//mlir::Value sz64 = mlir::arith::IndexCastUIOp::create(builder,loc, i64Type, data.getSizes()[0]);
			//mlir::Value sz192 = mlir::arith::ExtUIOp::create(builder,loc, ptrNsizeX2Int, sz64);
			//
			//mlir::Value str64 = mlir::arith::IndexCastUIOp::create(builder,loc, i64Type, data.getStrides()[0]);
			//mlir::Value str192 = mlir::arith::ExtUIOp::create(builder,loc, ptrNsizeX2Int, str64);
			//
			//mlir::Value ptr64 = mlir::arith::IndexCastUIOp::create(builder,loc, i64Type, ptr);
			//mlir::Value ptr192 = mlir::arith::ExtUIOp::create(builder,loc, ptrNsizeX2Int, ptr64);
			//
			//mlir::Value sizeShifted = mlir::arith::ShLIOp::create(builder,loc, sz192, c64);
			//mlir::Value offsetShifted = mlir::arith::ShLIOp::create(builder,loc, str192, c128);
			//
			//// OR them together
			//mlir::Value part1 = mlir::arith::OrIOp::create(builder,loc, ptr192, sizeShifted);
			//mlir::Value packed = mlir::arith::OrIOp::create(builder,loc, part1, offsetShifted);
			//
			//return packed;

			//return ref slice.
			return refSliceAlloc;
		},


			//Ignore these
		varcase(const AnyInvalidExpression auto&)->mlir::Value {
			throw std::runtime_error("Invalid expression type idx(" + std::to_string(itm.index()) + ") (mlir conversion)");
		}
		);
	}

	template<typename T>
	concept AnyIgnoredStat =
		std::same_as<T, parse::StatType::GotoV<true>>
		|| std::same_as<T, parse::StatType::Semicol>
		|| std::same_as<T, parse::StatType::Use>
		|| std::same_as<T, parse::StatType::FnDeclV<true>>
		|| std::same_as<T, parse::StatType::FunctionDeclV<true>>
		|| std::same_as<T, parse::StatType::DropV<true>>
		|| std::same_as<T, parse::StatType::ModV<true>>
		|| std::same_as<T, parse::StatType::ModAsV<true>>
		|| std::same_as<T, parse::StatType::UnsafeLabel>
		|| std::same_as<T, parse::StatType::SafeLabel>;


	inline GlobElemTy::Fn* getOrDeclFn(ConvData& conv,lang::MpItmId name,ast::Position place, const parse::ItmType::Fn* funcItmOrNull)
	{
		auto* mc = &conv.context;
		mlir::OpBuilder& builder = conv.builder;

		lang::MpItmId realName = name;
		{
			GlobElem* funcInfo = conv.getElement(realName);
			if (funcInfo != nullptr)
			{
				GlobElemTy::Fn* v = ezmatch(*funcInfo)(
				varcase(GlobElemTy::Alias)->GlobElemTy::Fn* {
					realName = var.realName;
					return nullptr;
				},
				varcase(GlobElemTy::Fn&) {
					return &var;
				}
				);
				if (v != nullptr)
					return v;

				funcInfo = conv.getElement(realName);
				return &std::get<GlobElemTy::Fn>(*funcInfo);
			}
		}

		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPointToStart(conv.module.getBody());

		if (funcItmOrNull == nullptr)
		{
			realName = parse::resolveAlias(conv.sharedDb, realName);
			if (realName != name)
			{
				GlobElem* funcInfo = conv.getElement(realName);
				if (funcInfo != nullptr)
					return &std::get<GlobElemTy::Fn>(*funcInfo);//todo: require it to be fn once other stuff is added.
			}
			funcItmOrNull = &parse::getItm<parse::ItmType::Fn>(
				conv.sharedDb, realName
			);
		}
		const parse::ItmType::Fn& funcItm = *funcItmOrNull;

		auto mangledName = mangleFuncName(conv, funcItm.abi, realName);

		llvm::SmallVector<mlir::Type> argTypes;
		for (const parse::ResolvedType& i : funcItm.args)
		{
			if (i.size != 0)
				argTypes.push_back(convType(conv, i, funcItm.abi));
		}
		llvm::SmallVector<mlir::Type, 1> retTypes;
		if (funcItm.ret.size != 0)
		{
			if (funcItm.abi == "C"sv)
				retTypes.push_back(convType(conv, funcItm.ret, funcItm.abi));
			else//output by ref.
				argTypes.push_back(convType(conv, funcItm.ret, funcItm.abi));
		}

		mlir::func::FuncOp funcOp = mlir::func::FuncOp::create(builder,
			convPos(conv, place), mangledName.sv(),
			mlir::FunctionType::get(mc, argTypes, retTypes),
			getExportAttr(conv, false),
			nullptr, nullptr, false
		);
		if (name != realName)
			conv.addElem(name, realName);
		return conv.addElem(realName, funcOp, funcItm.abi);
	}
	// Returns if region isnt terminated.
	inline bool convBlock(ConvData& conv,const parse::BlockV<true>& itm)
	{
		mlir::OpBuilder& builder = conv.builder;
		for (const auto& i : itm.statList)
			convStat(conv, i);
		if (itm.retTy!=parse::RetType::NONE && !itm.retExprs.empty())
		{
			//maybe return something
			//TODO
			return false;
		}
		return true;
	}

	inline std::optional<mlir::Value> convSoeOrBlock(ConvData& conv, const parse::SoeOrBlockV<true>& itm)
	{
		return ezmatch(itm)(
			varcase(const parse::SoeType::Expr&)->std::optional<mlir::Value> {
				return convExpr(conv, var->place,var->data);
			},
			varcase(const parse::SoeType::BlockV<true>&)->std::optional<mlir::Value> {
				if(convBlock(conv, var))
					return (mlir::Value)nullptr;
				return std::nullopt;
			}
		);
	}
	inline void convStat(ConvData& conv, const parse::Stat& itm)
	{
		auto* mc = &conv.context;
		mlir::OpBuilder& builder = conv.builder;

		ezmatch(itm.data)(

			varcase(const auto&) {
			throw std::runtime_error("Unimplemented statement type idx(" + std::to_string(itm.data.index()) + ") (mlir conversion)");
		},
			varcase(const parse::StatType::CanonicLocal&) {
			mlir::Value alloc = convExpr(conv, var.value.place, var.value.data);

			conv.localsStack.back().values[var.name.v] = { alloc ,nullptr};
		},
			varcase(const parse::StatType::RepeatUntilV<true>&) {
			mlir::Type i1Type = builder.getI1Type();
			const mlir::Location loc = convPos(conv, itm.place);
			// Initial loop-carried value: run once → %keepGoing = true
			mlir::Value one = mlir::arith::ConstantIntOp::create(builder,loc, i1Type,1);

			auto whileOp = mlir::scf::WhileOp::create(builder,loc, mlir::TypeRange{ i1Type }, mlir::ValueRange{ one });

			auto condArg = whileOp.getBefore().addArgument(i1Type, loc);
			builder.setInsertionPointToStart(whileOp.getBeforeBody());
			mlir::scf::ConditionOp::create(builder,loc, condArg, mlir::ValueRange{ condArg });

			whileOp.getAfter().addArgument(i1Type, loc);
			builder.setInsertionPointToStart(whileOp.getAfterBody());

			convBlock(conv, var.bl);

			// repeat-until: stop if cond is true → so loop if NOT result
			mlir::Value continueLoop = mlir::arith::XOrIOp::create(builder,
				loc, convExpr(conv,var.cond.place, var.cond.data), one);

			mlir::scf::YieldOp::create(builder,convPos(conv, var.bl.end), mlir::ValueRange{ continueLoop });

			builder.setInsertionPointAfter(whileOp);
		},
			varcase(const parse::StatType::WhileV<true>&) {
			auto whileOp = mlir::scf::WhileOp::create(builder,convPos(conv,itm.place), mlir::TypeRange{}, mlir::ValueRange{});

			builder.setInsertionPointToStart(whileOp.getBeforeBody());
			mlir::scf::ConditionOp::create(builder,convPos(conv, var.cond.place), convExpr(conv,var.cond.place,var.cond.data), mlir::ValueRange{});

			builder.setInsertionPointToStart(whileOp.getAfterBody());
			if(convBlock(conv, var.bl))
				mlir::scf::YieldOp::create(builder,convPos(conv,var.bl.end));

			builder.setInsertionPointAfter(whileOp);
		},
		varcase(const parse::StatType::ExternBlockV<true>&) {
			for (const auto& i : var.stats)
				convStat(conv, i);
		},
		varcase(const parse::StatType::UnsafeBlockV<true>&) {
			for (const auto& i : var.stats)
				convStat(conv, i);
		},
		varcase(const parse::StatType::BlockV<true>&) {
			auto scopeOp = mlir::memref::AllocaScopeOp::create(builder, convPos(conv, itm.place), mlir::TypeRange{});

			builder.setInsertionPointToStart(scopeOp.getBody());
			convBlock(conv, var);
			builder.setInsertionPointAfter(scopeOp);
		},
			varcase(const parse::StatType::IfCondV<true>&) {
			
			std::vector<mlir::Block*> yieldBlocks;
			yieldBlocks.reserve(var.elseIfs.size() + (var.elseBlock.has_value() ? 1 : 0));
			size_t elIfIdx = 0;
			mlir::scf::IfOp firstOp;
			do {
				const parse::Expr* cond;
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

				auto op = mlir::scf::IfOp::create(builder,loc, mlir::TypeRange{},
					convExpr(conv,cond->place, cond->data), hasMore);

				if (elIfIdx == 0)
					firstOp = op;

				builder.setInsertionPointToStart(op.thenBlock());
				std::optional<mlir::Value> ret = convSoeOrBlock(conv, *bl);
				if(ret.has_value())
				{
					if (*ret == nullptr)
						mlir::scf::YieldOp::create(builder, convPos(conv, itm.place));
					else
						mlir::scf::YieldOp::create(builder, convPos(conv, itm.place), *ret);
				}

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
				mlir::scf::YieldOp::create(builder, convPos(conv, itm.place));//TODO: return something.
			}

			builder.setInsertionPointAfter(firstOp);
		},
			varcase(const parse::StatType::AssignV<true>&) {
			if (var.vars.size() != 1 || var.exprs.size() != 1)
				throw std::runtime_error("Unimplemented assign conv, vers.size or var.exprs != 1");


			mlir::Value memRef = convExpr(
				conv,itm.place, var.vars[0]
			);
			const mlir::Location loc = convPos(conv, itm.place);

			mlir::Value expr = convExpr(conv, var.exprs[0].place, var.exprs[0].data);
			//Expr is a memref, so copy the bits/bytes.
			mlir::memref::CopyOp::create(builder,
				loc, expr, memRef
			);

			//auto zeroIndex = mlir::arith::ConstantIndexOp::create(builder,loc, 0);
			//mlir::memref::StoreOp::create(builder,loc, expr, memRef, mlir::ValueRange{ zeroIndex }, false);

		},
		varcase(const parse::StatType::Call&) {

			auto name = std::get<parse::ExprType::GlobalV<true>>(var.v->data);
			GlobElemTy::Fn* funcInfo = getOrDeclFn(conv,name,itm.place,nullptr);

			auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(mc);
			//It cant be the other types, as they are alreadt desugared!
			auto& argList = std::get<parse::ArgsType::ExprList>(var.args);
			std::vector<mlir::Value> args;
			args.reserve(argList.size());
			for (const auto& i : argList)
				args.emplace_back(convExpr(conv,i.place, i.data));
			
			// %res = llvm.call @puts(%llvm_ptr) : (!llvm.ptr) -> i32
			//mlir::LLVM::CallOp::create(builder,
			//	convPos(conv, itm.place),
			//	funcInfo->func.getResultTypes(),
			//	mlir::SymbolRefAttr::get(mc, funcInfo->func.getSymName()),
			//	llvm::ArrayRef<mlir::Value>{args});

			mlir::func::CallOp::create(builder,
				convPos(conv, itm.place),
				funcInfo->func.getResultTypes(),
				mlir::SymbolRefAttr::get(mc, funcInfo->func.getSymName()),
				llvm::ArrayRef<mlir::Value>{args});

		},
		varcase(const parse::StatType::ConstV<true>&) {

			conv.addLocalStackItem(var.local2Mp.names.size());

			const auto str = "Hello world\0"sv;
			auto i8Type = builder.getI8Type();
			auto i64Type = builder.getI64Type();
			auto strType = mlir::MemRefType::get({ (int64_t)str.size() }, i8Type, {}, 0);
			
			auto tensorType = mlir::RankedTensorType::get({ (int64_t)str.size() }, i8Type);
			auto denseStr = mlir::DenseElementsAttr::get(tensorType,llvm::ArrayRef{ (const int8_t*)str.data(),str.size() });

			mlir::memref::GlobalOp::create(builder,
				convPos(conv, itm.place),
				llvm::StringRef{ ":>::hello_world::greeting"sv },
				/*sym_visibility=*/getExportAttr(conv, var.exported),
				strType,
				denseStr,
				/*constant=*/true,
				/*alignment=*/builder.getIntegerAttr(i64Type, 1)
			);

			conv.localsStack.pop_back();

		},
		varcase(const parse::StatType::Fn&) {

			mlir::OpBuilder::InsertionGuard guard(builder);
			builder.setInsertionPointToStart(conv.module.getBody());

			const parse::ItmType::Fn& funcItm = parse::getItm<parse::ItmType::Fn>(
				conv.sharedDb, var.name
			);
			const bool cAbi = funcItm.abi == "C"sv;

			GlobElemTy::Fn* funcInfo = getOrDeclFn(conv, var.name, itm.place, &funcItm);
			if(funcItm.exported)
				funcInfo->func.setPublic();

			// Build a function in mlir
			mlir::Block* entry = funcInfo->func.addEntryBlock();
			builder.setInsertionPointToStart(entry);
			mlir::Location loc = convPos(conv, itm.place);

			//Locals, arguments.
			conv.addLocalStackItem(var.func.local2Mp.names.size());
			LocalStackItm& localTop = conv.localsStack.back();
			for (size_t i = 0; i < var.func.local2Mp.types.size(); i++)
				localTop.values[i].ty = &var.func.local2Mp.types[i];

			for (size_t i = 0; i < funcItm.argLocals.size(); i++)
			{
				parse::LocalId id = funcItm.argLocals[i];
				mlir::Value val = funcInfo->func.getArgument((unsigned int)i);
				if (cAbi)
				{//C ABI: arguments are passed by value, so we need to allocate a memref for them.
					auto memrefType = mlir::MemRefType::get({ 1 }, val.getType());
					mlir::Value alloc = mlir::memref::AllocaOp::create(builder,loc, memrefType);
					mlir::Value index0 = mlir::arith::ConstantIndexOp::create(builder,loc, 0);
					mlir::memref::StoreOp::create(builder,loc, val, alloc, mlir::ValueRange{ index0 });
					val = alloc;
				}
				localTop.values[id.v] = {
					val,
					&funcItm.args[i]
				};
			}

			if(convBlock(conv, var.func.block))
				mlir::func::ReturnOp::create(builder,convPos(conv, var.func.block.end));

			conv.localsStack.pop_back();
		},


		//Ignore these
		varcase(const AnyIgnoredStat auto&) {}
		);
	}
	export void conv(ConvData& conv) {
		convStat(conv, *conv.stat);
	}
}
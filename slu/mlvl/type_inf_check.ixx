module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <slu/ext/CppMatch.hpp>
#include <slu/Panic.hpp>
export module slu.mlvl.type_inf_check;

import slu.big_int;
import slu.settings;
import slu.ast.enums;
import slu.ast.make;
import slu.ast.mp_data;
import slu.ast.state;
import slu.ast.state_decls;
import slu.ast.type;
import slu.lang.basic_state;
import slu.lang.mpc;
import slu.mlvl.resolve_type;
import slu.visit.empty;
import slu.visit.visit;

namespace slu::mlvl
{
	using TypeInfCheckCfg = decltype(parse::sluCommon);

	template<class T, bool IS_STR = false>
	inline bool sameCheck(
	    const T& itm, const parse::ResolvedType& useTy, const auto& onSame)
	{
		return ezmatch(useTy.base)(
		    varcase(const auto&) { return false; },
		    varcase(const parse::RawTypeKind::TypeError) {
			    return true; //poisioned, so pass forward.
		    },
		    varcase(const parse::RawTypeKind::Variant&) {
			    for (const auto& i : var->options)
			    {
				    if (sameCheck<T, IS_STR>(itm, i, onSame))
					    return true; //Yep atleast one item is a valid thing.
			    }
			    return false;
		    },
		    //if its string, then match for ref chain. (also dont use ref chain
		    //by default as that could be T)
		    varcase(const parse::Sel<IS_STR, parse::RawTypeKind::String,
		        parse::RawTypeKind::Ref>&) {
			    if constexpr (IS_STR)
			    { // Subtyping into &str / &char.
				    //TODO: check lifetime
				    if (var->refType != ast::UnOpType::REF)
					    return false;
				    auto optName = var->elem.getStructName();
				    if (!optName.has_value())
					    return false;
				    lang::MpItmId name = *optName;

				    return name == mpc::STD_STR
				        || name == mpc::STD_CHAR; //TODO: require 1 ch only
			    }
			    return false;
		    },
		    varcase(const T&) { return onSame(itm, var); });
	}
	inline bool rangeRangeSubtypeCheck(
	    const parse::AnyRawRange auto itm, const parse::AnyRawRange auto useTy)
	{
		return itm.max <= useTy.max && itm.min >= useTy.min;
	}
	inline bool _compilerHackGt(auto a, auto b)
	{
		return a > b;
	}
	template<class T1, class T2> inline bool _compilerHackEq(auto a, auto b)
	{
		return (T1)a == (T2)b;
	}
	template<parse::AnyRawIntOrRange T>
	inline bool intRangeSubtypeCheck(
	    const T itm, const parse::ResolvedType& useTy)
	{
		return ezmatch(useTy.base)(
		    varcase(const auto&) { return false; },
		    varcase(const parse::RawTypeKind::TypeError) {
			    return true; //poisioned, so pass forward.
		    },
		    varcase(const parse::RawTypeKind::Variant&) {
			    for (const auto& i : var->options)
			    {
				    if (intRangeSubtypeCheck(itm, i))
					    return true; //Yep atleast one item is a valid thing.
			    }
			    return false;
		    },
		    varcase(const parse::AnyRawIntOrRange auto&) {
			    using VarT = std::remove_cvref_t<decltype(var)>;
			    constexpr bool itmIsInt = parse::AnyRawInt<T>;
			    constexpr bool isInt = parse::AnyRawInt<VarT>;

			    if constexpr (itmIsInt && isInt)
			    { // Check for sign mismatch
				    if constexpr (std::same_as<T, parse::RawTypeKind::Int64>
				        && std::same_as<VarT, parse::RawTypeKind::Uint64>)
				    {
					    if (var > (uint64_t)INT64_MAX)
						    return false;
					    return _compilerHackEq<T, int64_t>(itm, var);
				    } else if constexpr (std::same_as<T,
				                             parse::RawTypeKind::Uint64>
				        && std::same_as<VarT, parse::RawTypeKind::Int64>)
				    {
					    if (_compilerHackGt(itm, (uint64_t)INT64_MAX))
						    return false;
					    return _compilerHackEq<int64_t, VarT>(itm, var);
				    } else
					    return itm == var;
			    } else if constexpr (itmIsInt)
				    return var.isInside(itm);
			    else if constexpr (isInt)
				    return itm.isOnly(var);
			    else
				    return rangeRangeSubtypeCheck(itm, var);
		    });
	}
	bool subtypeCheck(const parse::BasicMpDbData& mpDb,
	    const parse::ResolvedType& subty, const parse::ResolvedType& useTy);

	inline bool nameMatchCheck(const parse::BasicMpDbData& mpDb,
	    lang::MpItmId subName, lang::MpItmId useName)
	{
		if (subName == useName)
			return true; //Same name, so match.
		if (subName.empty())
			return true;
		if (useName.empty())
			return false; //Named -/> unnamed.
		//TODO: upcasting for enums.
		//TODO: unhardcode this maybe? or keep it for perf reasons?
		if ((subName == mpc::STD_BOOL_TRUE || subName == mpc::STD_BOOL_FALSE)
		    && useName == mpc::STD_BOOL)
			return true; //allow true/false -> bool

		return false;
	}
	template<class T>
	inline bool nearExactCheckDeref(
	    const T& subty, const parse::ResolvedType& useTy)
	{
		if (!std::holds_alternative<T>(useTy.base))
			return false;
		return subty->nearlyExact(*std::get<T>(useTy.base));
	}

	inline bool subtypeCheckRef(
	    const parse::RawTypeKind::Ref& var, const parse::ResolvedType& useTy)
	{ //TODO: lifetimes
		using T = parse::RawTypeKind::Ref;
		return sameCheck<T>(var, useTy, [&](const T& var, const T& useTy) {
			if (var->refType != useTy->refType || var->life != useTy->life)
				return false;
			return nearExactCheck(var->elem, useTy->elem);
		});
	}
	inline bool subtypeCheckPtr(
	    const parse::RawTypeKind::Ptr& var, const parse::ResolvedType& useTy)
	{
		using T = parse::RawTypeKind::Ptr;
		return sameCheck<T>(var, useTy, [&](const T& var, const T& useTy) {
			if (var->ptrType != useTy->ptrType)
				return false;
			return nearExactCheck(var->elem, useTy->elem);
		});
	}
	inline bool subtypeCheckSlice(
	    const parse::RawTypeKind::Slice& var, const parse::ResolvedType& useTy)
	{
		using T = parse::RawTypeKind::Slice;
		return sameCheck<T>(var, useTy, [&](const T& var, const T& useTy) {
			return nearExactCheck(var->elem, useTy->elem);
		});
	}

	inline bool subtypeCheck(const parse::BasicMpDbData& mpDb,
	    const parse::ResolvedType& subty, const parse::ResolvedType& useTy)
	{
		return ezmatch(subty.base)(
		    varcase(const parse::RawTypeKind::Unresolved&)->bool {
			    throw std::runtime_error(
			        "Found unresolved type in subtype check");
		    },
		    varcase(const parse::RawTypeKind::Inferred)->bool {
			    throw std::runtime_error(
			        "Found inferred type in subtype check");
		    },
		    varcase(const parse::RawTypeKind::TypeError) {
			    return true; //poisioned, so pass forward.
		    },
		    varcase(const parse::RawTypeKind::String&) {
			    using T = parse::RawTypeKind::String;
			    return sameCheck<T, true>(var, useTy, std::equal_to<T>{});
		    },
		    varcase(const parse::RawTypeKind::Float64) {
			    using T = parse::RawTypeKind::Float64;
			    return sameCheck<T>(var, useTy,
			        std::equal_to<T>{}); //TODO: allow f64 as the type too
			                             //(needs to be impl first).
		    },

		    varcase(const parse::RawTypeKind::Variant&) {
			    for (const auto& i : var->options)
			    {
				    if (!subtypeCheck(mpDb, i, useTy))
					    return false;
			    }
			    return true;
		    },
		    varcase(const parse::RawTypeKind::Union&) {
			    using T = parse::RawTypeKind::Union;
			    return sameCheck<T>(
			        var, useTy, [&](const T& var, const T& useTy) {
				        if (var->fields.size() != var->fields.size())
					        return false;
				        if (!nameMatchCheck(mpDb, var->name, useTy->name))
					        return false;

				        for (size_t i = 0; i < var->fields.size(); i++)
				        {
					        if (!nearExactCheck(
					                var->fields[i], useTy->fields[i]))
						        return false;
					        if (var->fieldNames[i] != useTy->fieldNames[i])
						        return false;
				        }
				        return true;
			        });
		    },
		    varcase(const parse::RawTypeKind::Struct&) {
			    using T = parse::RawTypeKind::Struct;
			    return sameCheck<T>(
			        var, useTy, [&](const T& var, const T& useTy) {
				        if (!nameMatchCheck(mpDb, var->name, useTy->name))
					        return false;
				        size_t otherIdx = 0;
				        for (size_t i = 0; i < var->fields.size(); i++)
				        {
					        const parse::ResolvedType& ty = var->fields[i];
					        const std::string& name = var->fieldNames[i];
					        bool exit = true;
					        //locate same field in other type & subtype check
					        //it.
					        for (; otherIdx < useTy->fieldNames.size();
					            otherIdx++)
					        {
						        if (name != useTy->fieldNames[otherIdx])
							        continue;
						        if (!subtypeCheck(
						                mpDb, ty, useTy->fields[otherIdx]))
							        return false;
						        exit = false;
					        }
					        if (exit)
						        return false;
				        }
				        return true;
			        });
		    },

		    varcase(const parse::RawTypeKind::Ref&) {
			    return subtypeCheckRef(var, useTy);
		    },
		    varcase(const parse::RawTypeKind::Ptr&) {
			    return subtypeCheckPtr(var, useTy);
		    },
		    varcase(const parse::RawTypeKind::Slice&) {
			    return subtypeCheckSlice(var, useTy);
		    },

		    varcase(const parse::AnyRawIntOrRange auto&) {
			    return intRangeSubtypeCheck(var, useTy);
		    });
	}

	extern "C++"
	{
	//ignores outerSliceDims of either side, as if checking the slices element
	//type.
	bool nearExactCheck(const slu::parse::ResolvedType& subty,
	    const slu::parse::ResolvedType& useTy)
	{
		if (std::holds_alternative<parse::RawTypeKind::TypeError>(useTy.base))
			return true; //poisioned, so pass forward.

		//This should already be true, if all parts of the near exact check are
		//correct. if (subty.size != useTy.size) 	return false;
		if (subty.alignmentData != useTy.alignmentData)
			return false;

		return ezmatch(subty.base)(
		    [&]<class T>(const T& var) {
			    if (!std::holds_alternative<T>(useTy.base))
				    return false;
			    return var == std::get<T>(useTy.base);
		    },
		    varcase(const parse::RawTypeKind::TypeError) { return true; },
		    varcase(const parse::RawTypeKind::Inferred)->bool {
			    throw std::runtime_error(
			        "TODO: error logging, Found Inferred type in near exact type check");
		    },
		    varcase(const parse::RawTypeKind::Unresolved&)->bool {
			    throw std::runtime_error(
			        "TODO: error logging, Found unresolved type in near exact type check");
		    },

		    varcase(const parse::RawTypeKind::Variant&) {
			    return nearExactCheckDeref(var, useTy);
		    },
		    varcase(const parse::RawTypeKind::Struct&) {
			    using T = parse::RawTypeKind::Struct;
			    if (!std::holds_alternative<T>(useTy.base))
				    return false;
			    if (nearExactCheckDeref(var, useTy))
				    return true;

			    return false;
		    },
		    varcase(const parse::RawTypeKind::Union&) {
			    return nearExactCheckDeref(var, useTy);
		    },

		    varcase(const parse::RawTypeKind::Ref&) {
			    return subtypeCheckRef(var, useTy);
		    },
		    varcase(const parse::RawTypeKind::Ptr&) {
			    return subtypeCheckPtr(var, useTy);
		    },
		    varcase(const parse::RawTypeKind::Slice&) {
			    return subtypeCheckSlice(var, useTy);
		    });
	}
	}


	using TmpVar = uint64_t;
	using VisitTypeBuilder = std::variant<parse::ResolvedType,
	    const parse::ResolvedType*, parse::LocalId, TmpVar>;

	struct SubTySide
	{
		std::vector<TmpVar> tmpLocals;
		std::vector<parse::LocalId> locals;
		std::vector<parse::ResolvedType> tys;
		std::vector<const parse::ResolvedType*> tyRefs;

		void clear()
		{
			tmpLocals.clear();
			locals.clear();
			tys.clear();
			tyRefs.clear();
		}
	};
	struct MethodCall
	{
		std::vector<TmpVar> args;
		lang::MpItmId* name;
		TmpVar selfArg;
	};
	struct FieldGet
	{
		lang::PoolString name;
		TmpVar selfArg;
	};
	struct LocalVarInfo
	{
		parse::ResolvedType* resolvedType = nullptr;

		//std::vector<lang::LocalObjId> usedFields;
		//std::vector<lang::MpItmId> usedMethods;
		//traits?
		//???

		SubTySide use; //Requirements when used.
		std::variant<SubTySide, MethodCall, FieldGet>
		    edit; //Requirements when writen to.

		bool boolLike : 1 = false; //part of use.
		bool taken    : 1 = false;
		bool resolved : 1 = false;

		void resolveNoCheck(parse::ResolvedType&& t)
		{
			Slu_assert(!resolved);
			//use.clear(); //Dont, because we check use-types after calling this
			//func usedFields.clear(); usedMethods.clear();
			edit = {};
			resolved = true;
			if (resolvedType != nullptr)
				*resolvedType = std::move(t);
		}
		void requireBoolLike()
		{
			if (resolved && !boolLike && !resolvedType->isBool())
				throw std::runtime_error(
				    "TODO: error logging, found non bool expr");

			boolLike = true;
		}
	};
	using LocalVarList = std::vector<LocalVarInfo>;

	struct BreakPoint
	{
		lang::PoolString name; //empty -> not named.
		TmpVar out;            // size-max -> scope limiter.

		static constexpr BreakPoint newScopeLimiter()
		{
			return BreakPoint{lang::PoolString::newEmpty(), SIZE_MAX};
		}

		constexpr bool scopeLimiter() const
		{
			return out == SIZE_MAX;
		}
	};

	struct TypeInfCheckVisitor : visit::EmptyVisitor<TypeInfCheckCfg>
	{
		using Cfg = TypeInfCheckCfg;
		static constexpr bool isSlu = true;

		const parse::BasicMpDbData& mpDb;
		std::vector<parse::Locals<Cfg>*> localsStack;
		std::vector<LocalVarList> localsDataStack;
		std::vector<LocalVarList> tmpLocalsDataStack;

		std::vector<VisitTypeBuilder> exprTypeStack;
		std::vector<BreakPoint> breakPointStack;

		LocalVarInfo& localVar(const parse::LocalId id)
		{
			return localsDataStack.back()[id.v];
		}
		LocalVarInfo& localVar(const TmpVar id)
		{
			return tmpLocalsDataStack.back()[id];
		}

		void handleVisTyBuilder(
		    const VisitTypeBuilder& t, auto&& appRes, auto&& appLocal)
		{
			ezmatch(t)(
			    varcase(const parse::ResolvedType&) { appRes(var); },
			    varcase(const parse::ResolvedType*) { appRes(*var); },
			    varcase(const parse::LocalId) { appLocal(localVar(var)); },
			    varcase(const TmpVar) { appLocal(localVar(var)); });
		}

		void requireAsBool(const VisitTypeBuilder& t)
		{
			handleVisTyBuilder(
			    t,
			    [&](const parse::ResolvedType& var) {
				    if (!var.isBool())
					    throw std::runtime_error(
					        "TODO: error logging, found non bool expr");
			    },
			    [&](LocalVarInfo& var) { var.requireBoolLike(); });
		}
		void requireUseTy(
		    const VisitTypeBuilder& t, const parse::ResolvedType& ty)
		{
			handleVisTyBuilder(
			    t,
			    [&](const parse::ResolvedType& var) {
				    if (!subtypeCheck(mpDb, var, ty))
					    throw std::runtime_error(
					        "TODO: error logging, found non matching type expr");
			    },
			    [&](LocalVarInfo& var) { var.use.tyRefs.push_back(&ty); });
		}

		template<class RawT> bool handleConstType(auto&& v)
		{
			exprTypeStack.emplace_back(
			    parse::ResolvedType::getConstType(RawT{std::move(v)}));
			return false;
		}
		template<class T>
		void editLocalVar(T itm)
		    requires(std::same_as<T, parse::LocalId> || std::same_as<T, TmpVar>)
		{
			VisitTypeBuilder& editTy = exprTypeStack.back();
			handleVisTyBuilder(
			    editTy, [&](const parse::ResolvedType& var) {},
			    [&](LocalVarInfo& var) {
				    if constexpr (std::same_as<T, TmpVar>)
					    var.use.tmpLocals.push_back(itm);
				    else
					    var.use.locals.push_back(itm);
			    });

			ezmatch(editTy)(
			    varcase(parse::ResolvedType&) {
				    std::get<SubTySide>(localVar(itm).edit)
				        .tys.emplace_back(std::move(var));
			    },
			    varcase(const parse::ResolvedType*) {
				    std::get<SubTySide>(localVar(itm).edit)
				        .tyRefs.emplace_back(var);
			    },
			    varcase(const parse::LocalId) {
				    std::get<SubTySide>(localVar(itm).edit)
				        .locals.emplace_back(var);
			    },
			    varcase(const TmpVar) {
				    std::get<SubTySide>(localVar(itm).edit)
				        .tmpLocals.emplace_back(var);
			    });
			exprTypeStack.pop_back();
		}


		bool preExpr(parse::Expr& itm)
		{
			exprTypeStack.emplace_back();
			return false;
		}

		bool preInferr(parse::ExprType::Infer itm)
		{
			throw std::runtime_error(
			    "TODO: type-check/inferr type of '?', it might be a error or just 'type'.");
		}
		bool preOpenRange(parse::ExprType::OpenRange itm)
		{
			throw std::runtime_error(
			    "TODO: type-check/inferr open range expressions.");
		}
		bool preF64(parse::ExprType::F64 itm)
		{
			return handleConstType<parse::RawTypeKind::Float64>(itm);
		}
		bool preI64(parse::ExprType::I64 itm)
		{
			return handleConstType<parse::RawTypeKind::Int64>(itm);
		}
		bool preU64(parse::ExprType::U64 itm)
		{
			return handleConstType<parse::RawTypeKind::Uint64>(itm);
		}
		bool preM128(parse::ExprType::M128 itm)
		{
			return handleConstType<parse::RawTypeKind::Neg128>(itm);
		}
		bool preP128(parse::ExprType::P128 itm)
		{
			return handleConstType<parse::RawTypeKind::Pos128>(itm);
		}
		bool preExprString(parse::ExprType::String& itm)
		{
			return handleConstType<parse::RawTypeKind::String>(
			    std::string(itm.v));
		}
		bool preLocalExpr(const parse::ExprType::Local itm)
		{
			exprTypeStack.back() = itm;
			return false;
		}
		bool preGlobalExpr(parse::ExprType::Global<Cfg>& itm)
		{
			//TODO
			return false;
		}
		bool preTableExpr(parse::ExprType::Table<Cfg>& itm)
		{
			//TODO
			return true;
		}
		//TODO: create TmpVar's for (self)call/indexing/deref results, when the
		//types are not obvious.
		void postDerefExpr(parse::ExprType::Deref& itm)
		{
			//TODO: could be deref or deref-mut or ... lol
		}
		bool preIndexExpr(parse::ExprType::Index& itm)
		{
			//TODO: need to finalize spec first?
			return true;
		}
		void postFieldExpr(parse::ExprType::Field<Cfg>& itm)
		{
			if (itm.field == mpDb.getPoolStr("__convHack__refSlice_ptr"))
			{
				auto& tmpVars = tmpLocalsDataStack.back();
				TmpVar varId = tmpVars.size();
				LocalVarInfo& var = tmpVars.emplace_back(&itm.ty);
				std::get<SubTySide>(var.edit).tys.emplace_back(
				    parse::PtrRawType::newTy(
				        parse::ResolvedType::newU8(), ast::UnOpType::PTR));
				exprTypeStack.back() = varId;
				return;
			}
			visit::visitExpr(*this, *itm.v);

			auto& tmpVars = tmpLocalsDataStack.back();
			TmpVar selfId = tmpVars.size();
			tmpVars.emplace_back(nullptr);
			editLocalVar(selfId);

			TmpVar resId = tmpVars.size();
			tmpVars.emplace_back(&itm.ty).edit
			    = FieldGet{.name = itm.field, .selfArg = resId};
		}
		bool preCallExpr(parse::ExprType::Call& itm)
		{
			handleCall<false, true>(itm);
			return true;
		}
		void checkTypeMethod(const parse::ResolvedType& var,
		    const parse::Args& args, lang::MpItmId& method,
		    parse::ResolvedType& resTy)
		{
			//TODO: check if var impls that method & what it is

			//TODO: "Complex" method resolution.

			//TODO: if method is a trait-fn, then: ???
			//TODO: impl/builtin-impl found for method

			auto& argList = std::get<parse::ArgsType::ExprList>(args);
			//TODO: check arg use types
		}
		bool preSelfCallExpr(parse::ExprType::SelfCall& itm)
		{
			visit::visitExpr(*this, *itm.v);
			auto& tmpVars = tmpLocalsDataStack.back();

			TmpVar selfId = tmpVars.size();
			tmpVars.emplace_back(
			    nullptr); //TODO: special handling of not outputing
			editLocalVar(selfId);
			//var.usedMethods.emplace_back(itm.method);
			//Tmp var for each arg.
			//tmp var for output

			auto& argList = std::get<parse::ArgsType::ExprList>(itm.args);
			std::vector<TmpVar> args;
			for (auto& i : argList)
			{
				//Make temp var for func result, also add editType for it.
				visit::visitExpr(*this, i);
				auto& tmpVars
				    = tmpLocalsDataStack.back(); //visitExpr could realloc it

				TmpVar varId = tmpVars.size();
				tmpVars.emplace_back(
				    nullptr); //TODO: special handling of not outputing
				editLocalVar(varId);
				args.emplace_back(varId);
			}
			auto& tmpVars2 = tmpLocalsDataStack.back(); //may realloc

			//Make temp var for func result, also add editType for it.
			TmpVar varId = tmpVars2.size();
			tmpVars2.emplace_back(&itm.ty).edit
			    = MethodCall(std::move(args), &itm.method, selfId);

			exprTypeStack.back() = varId;

			return true;
		}

		//

		template<bool voidOutput, bool boxed>
		void handleCall(parse::Call<boxed>& itm)
		{
			if (!std::holds_alternative<parse::ArgsType::ExprList>(itm.args))
				throw std::runtime_error(
				    "TODO: type inference for func call with complex args.");

			if (!std::holds_alternative<parse::ExprType::Global<Cfg>>(
			        itm.v->data))
				throw std::runtime_error(
				    "TODO: type inference for func call on non-global var.");

			const parse::ItmType::Fn& funcItm
			    = parse::getItm<parse::ItmType::Fn, true>(
			        mpDb, std::get<parse::ExprType::Global<Cfg>>(itm.v->data));

			parse::ArgsType::ExprList& args
			    = std::get<parse::ArgsType::ExprList>(itm.args);
			//Restrict arg exprs to match types in funcItm.
			for (size_t i = args.size(); i > 0; i++)
			{
				visit::visitExpr(*this, args[i]);
				const parse::ResolvedType& ty = funcItm.args[i];
				requireUseTy(exprTypeStack.back(), ty);
				exprTypeStack.pop_back();
			}
			if constexpr (!voidOutput)
			{
				auto& tmpVars = tmpLocalsDataStack.back();
				//Make temp var for func result, also add editType for it.
				TmpVar varId = tmpVars.size();
				auto& someEdit = tmpVars.emplace_back(&itm.ty).edit;
				someEdit = SubTySide();
				std::get<SubTySide>(someEdit).tyRefs.emplace_back(&funcItm.ret);

				exprTypeStack.back() = varId;
			}
		}

		//Restrictions.
		void postAnyCond(parse::Expr& itm)
		{
			requireAsBool(exprTypeStack.back());
			exprTypeStack.pop_back();
		}
		void postCanonicLocal(parse::StatType::CanonicLocal& itm)
		{
			editLocalVar(itm.name); //TODO: restrict the type to exactly that?
			                        //(unless it is inferr)
		}
		bool preCallStat(parse::StatType::Call& itm)
		{
			handleCall<true, false>(itm);
			return true;
		}
		void handleSoe(TmpVar resId, parse::SoeOrBlock<Cfg>& itm)
		{
			ezmatch(itm)(
			    varcase(parse::SoeType::Expr&) {
				    visit::visitExpr(*this, *var);
				    editLocalVar(resId);
			    },
			    varcase(parse::SoeType::Block<Cfg>&) {
				    visit::visitBlock(*this, var);
			    });
		}
		template<bool isExpr>
		bool handleIfCond(parse::BaseIfCond<Cfg, isExpr>& itm)
		{
			auto& tmpVars = tmpLocalsDataStack.back();
			TmpVar resId = tmpVars.size();
			tmpVars.emplace_back(&itm.ty);

			visit::visitExpr(*this, *itm.cond);
			requireAsBool(exprTypeStack.back());
			exprTypeStack.pop_back();

			handleSoe(resId, *itm.bl);
			for (auto& i : itm.elseIfs)
			{
				visit::visitExpr(*this, i.first);
				requireAsBool(exprTypeStack.back());
				exprTypeStack.pop_back();
				handleSoe(resId, i.second);
			}
			handleSoe(resId, **itm.elseBlock);

			if constexpr (isExpr)
				exprTypeStack.back() = resId;
			return true;
		}
		bool preIfExpr(parse::ExprType::IfCond<Cfg>& itm)
		{
			return handleIfCond<true>(itm);
		}
		bool preIfStat(parse::StatType::IfCond<Cfg>& itm)
		{
			return handleIfCond<false>(itm);
		}
		bool preAssign(parse::StatType::Assign<Cfg>& itm)
		{
			for (size_t i = 0; i < itm.vars.size(); i++)
			{
				parse::ExprData<Cfg>& var = itm.vars[i];
				visit::visitExpr(*this, itm.exprs[i]);

				ezmatch(var)(
				    varcase(auto&) {
					    throw std::runtime_error(
					        "TODO: type inference/checking for any expr in assign statement.");
				    },
				    varcase(parse::ExprType::GlobalV<true>&) {
					    throw std::runtime_error(
					        "TODO: type check global assign statement.");
				    },
				    varcase(parse::ExprType::Local&) { editLocalVar(var); });
			}
			return true;
		}

		//Allow any type.
		void postDrop(parse::StatType::Drop<Cfg>&)
		{
			exprTypeStack.pop_back();
		}

		//Ignored.
		bool preCanonicGlobal(parse::StatType::CanonicGlobal&)
		{
			return true;
		}
		bool preTypeExpr(parse::Expr&)
		{
			return true;
		}
		//Stack stuff.
		bool preLocals(parse::Locals<Cfg>& itm)
		{
			localsStack.push_back(&itm);

			localsDataStack.emplace_back();
			localsDataStack.back().resize(itm.names.size());

			itm.types.resize(itm.names.size());
			for (size_t i = 0; i < itm.names.size(); i++)
				localsDataStack.back()[i].resolvedType = &itm.types[i];

			tmpLocalsDataStack.emplace_back();
			if (!breakPointStack.empty())
				breakPointStack.emplace_back(BreakPoint::newScopeLimiter());
			return false;
		}

		void visitSubTySide(SubTySide& side, auto&& visitor)
		{
			for (auto& i : side.tmpLocals)
				visitor(resolveLocal(localVar(i)));
			for (auto& i : side.locals)
				visitor(resolveLocal(localVar(i)));
			for (auto& i : side.tys)
				visitor(i);
			for (auto& i : side.tyRefs)
				visitor(*i);
		}

		// scales: O(N^3) // n(1+2n^2+3n)/6 // n is number of
		// assignments*avg1OrVariantSize.
		void addSubTypeToList(std::vector<const parse::ResolvedType*>& types,
		    const parse::ResolvedType& editTy)
		{
			for (const parse::ResolvedType* j : types)
			{
				if (subtypeCheck(mpDb, editTy, *j))
					return; //Already found a supertype of it.
			}
			types.emplace_back(&editTy);
			//TODO: maybe look for subtypes of editTy, or atleast unify them.
		}
		void visitTypeForInference(bool& poison,
		    std::vector<const parse::ResolvedType*>& types,
		    std::vector<parse::Range129>& intRanges,
		    const parse::ResolvedType& editTy)
		{
			if (!editTy.isComplete())
				throw std::runtime_error(
				    "TODO: error logging, found incomplete type expr");
			ezmatch(editTy.base)(
			    varcase(const parse::RawTypeKind::Unresolved&) {
				    throw std::runtime_error(
				        "TODO: error logging, found unresolved type expr");
			    },
			    varcase(const parse::RawTypeKind::Inferred) {
				    throw std::runtime_error(
				        "TODO: error logging, found inferred type expr");
			    },
			    varcase(const parse::RawTypeKind::TypeError) { poison = true; },
			    varcase(const parse::RawTypeKind::String&) {
				    addSubTypeToList(types, editTy);
			    },
			    varcase(const parse::RawTypeKind::Union&) {
				    addSubTypeToList(types, editTy);
			    },
			    varcase(const parse::RawTypeKind::Struct&) {
				    addSubTypeToList(
				        types, editTy); //Todo: potential unification.
			    },
			    varcase(const parse::RawTypeKind::Variant&) {
				    for (auto& j : var->options)
					    visitTypeForInference(poison, types, intRanges, j);
			    },
			    varcase(const parse::RawTypeKind::Ref&) {
				    addSubTypeToList(types, editTy);
			    },
			    varcase(const parse::RawTypeKind::Ptr&) {
				    addSubTypeToList(types, editTy);
			    },
			    varcase(const parse::RawTypeKind::Slice&) {
				    addSubTypeToList(types, editTy);
			    },

			    varcase(const parse::RawTypeKind::Float64){
			        //TODO: float+float -> float-type... or float+float -> float
			        //|| float?
			    },
			    [&]<parse::AnyRawIntOrRange T>(const T& var) {
				    if constexpr (parse::AnyRawInt<T>)
					    intRanges.emplace_back(parse::range129FromInt(var));
				    else if constexpr (std::same_as<T,
				                           parse::RawTypeKind::Range64>)
					    intRanges.emplace_back(parse::range129From64(var));
				    else
					    intRanges.emplace_back(var);
			    });
		}

		//Returns if failed
		bool resolveMethod(
		    std::vector<parse::ResolvedType>& derefStack, lang::MpItmId& name)
		{
			if (derefStack.size() > 16) //TODO: config for limit
				return true;            //TODO: error

			//if(name==mpc::STD_OPS_ADD_ADD)

			//TODO:
			return false;
		}

		void checkLocals(std::span<LocalVarInfo> locals)
		{
			for (LocalVarInfo& i : locals)
			{
				if (i.resolved)
					continue;
				if (i.taken)
					throw std::runtime_error(
					    "TODO: error logging, variable type depends on itself, cant inferr it");
				i.taken = true;
				// Resolve its type

				if (i.boolLike)
				{ // "true", "false" or "bool"
					bool canTrue = false;
					bool canFalse = false;
					bool poison = false;

					//First resolve those things & also check types a bit.

					ezmatch(i.edit)(
					    varcase(MethodCall&){
					        //TODO: method call support
					    },
					    varcase(const FieldGet){
					        //TODO: field get support
					    },
					    varcase(SubTySide&) {
						    visitSubTySide(var, [&](const parse::ResolvedType& otherTy) {
							    if (std::holds_alternative<
							            parse::RawTypeKind::TypeError>(
							            otherTy.base))
							    {
								    poison = true;
								    return;
							    }

							    if (!otherTy.isComplete())
								    throw std::runtime_error(
								        "TODO: error logging, found incomplete type expr");
							    if (otherTy.size > 1)
								    throw std::runtime_error(
								        "TODO: error logging, found non bool expr");
							    auto tyNameOpt = otherTy.getStructName();
							    if (!tyNameOpt)
								    throw std::runtime_error(
								        "TODO: error logging, found non bool expr");
							    lang::MpItmId tyName = *tyNameOpt;
							    if (tyName == mpc::STD_BOOL)
								    canTrue = canFalse = true;
							    else if (tyName == mpc::STD_BOOL_TRUE)
								    canTrue = true;
							    else if (tyName == mpc::STD_BOOL_FALSE)
								    canFalse = true;
							    else
								    throw std::runtime_error(
								        "TODO: error logging, found non bool expr");
						    });
					    });

					if (!(canTrue || canFalse))
					{
						poison = true;
						throw std::runtime_error(
						    "TODO: error logging, found non bool type");
					}

					if (poison)
						i.resolveNoCheck(parse::ResolvedType::newError());
					else
						i.resolveNoCheck(
						    parse::StructRawType::newBoolTy(canTrue, canFalse));
				} else
				{ //resolve it (find a type, that is a subtype of all the edit
				  //types).
					bool poison = false;
					std::vector<const parse::ResolvedType*> types;
					std::vector<parse::Range129> intRanges;
					bool alreadyResolved = false;

					ezmatch(i.edit)(
					    varcase(MethodCall&) {
						    std::vector<parse::ResolvedType> derefStack(1);

						    LocalVarInfo& selfInfo = localVar(var.selfArg);
						    selfInfo.resolvedType = &derefStack.front();
						    checkLocals(std::span(&selfInfo, 1));
						    //Now that self is known, we need to know the
						    //method.

						    if (resolveMethod(derefStack, *var.name))
						    {
							    poison = true;
							    for (const TmpVar k : var.args)
								    localVar(k).resolveNoCheck(
								        parse::ResolvedType::newError());
							    return;
						    }

						    //we resolved this type, so now we resolve & check
						    //the args
						    for (const TmpVar k : var.args)
						    {
							    //TODO: add use ty with the methods real type
							    checkLocals(std::span(&localVar(k), 1));
						    }
						    //Now that args are of known types, we need to know
						    //what the result is!
						    //TODO: output the result type into types array
					    },
					    varcase(const FieldGet) {
						    LocalVarInfo& selfInfo = localVar(var.selfArg);
						    parse::ResolvedType tmpTy;
						    if (selfInfo.resolvedType == nullptr)
							    selfInfo.resolvedType = &tmpTy;
						    checkLocals(std::span(&selfInfo, 1));

						    parse::ResolvedType& ty = *selfInfo.resolvedType;
						    // check if it even has that field
						    std::string_view fieldView = mpDb.getSv(var.name);
						    //TODO: auto deref
						    ezmatch(ty.base)(
						        varcase(auto&) {
							        poison = true;
							        //TODO: msg about trying to field into
							        //something with no fields.
						        },
						        varcase(parse::RawTypeKind::TypeError) {
							        poison = true;
						        },
						        [&]<class T>(T& var2)
						            requires std::same_as<T,
						                         parse::RawTypeKind::Struct>
						        || std::same_as<T, parse::RawTypeKind::Union>
						        {
							        size_t idx = 0;
							        bool good = false;
							        for (const std::string& j :
							            var2->fieldNames)
							        {
								        if (j == fieldView)
								        {
									        good = true;
									        break;
								        }
								        idx++;
							        }
							        if (!good)
							        {
								        poison = true;
								        //TODO: struct doesnt have that field.
								        return;
							        }
							        if (&ty == &tmpTy)
							        {
								        alreadyResolved = true;
								        i.resolveNoCheck(
								            std::move(var2->fields[idx]));
							        } else
								        types.push_back(&var2->fields[idx]);
						        });
					    },
					    varcase(SubTySide&) {
						    visitSubTySide(
						        var, [&](const parse::ResolvedType& editTy) {
							        visitTypeForInference(
							            poison, types, intRanges, editTy);
						        });
					    });


					if (poison)
						i.resolveNoCheck(parse::ResolvedType::newError());
					else if (!alreadyResolved)
					{
						parse::Range129 fullIntRange;
						if (!intRanges.empty())
						{
							// Sort by start, end doesnt matter, cuz we only
							// care about overlapping/adjacency.
							std::sort(intRanges.begin(), intRanges.end(),
							    [&](const parse::Range129& a,
							        const parse::Range129& b) {
								    return parse::r129Get(
								        a, [&](const auto& aRange) {
									        return parse::r129Get(
									            b, [&](const auto& bRange) {
										            return aRange.min
										                < bRange.min;
									            });
								        });
							    });
							bool first = true;
							for (const auto& j : intRanges)
							{
								if (first)
								{
									fullIntRange = j;
									continue;
								}
								parse::r129Get(j, [&](const auto& jRange) {
									parse::r129Get(fullIntRange,
									    [&](const auto& fullRange) {
										    if (jRange.min.lteOtherPlus1(
										            fullRange.max))
										    { // Overlapping or adjacent, so
											  // merge them.
											    const bool maxBigger
											        = fullRange.max
											        < jRange.max;
											    const bool minSmaller
											        = fullRange.min
											        > jRange.min;

											    if (maxBigger && minSmaller)
												    fullIntRange = j;
											    else
											    {
												    if constexpr (
												        !(std::same_as<
												              decltype(fullRange
												                      .min),
												              Integer128<false,
												                  false>>
												            && std::same_as<
												                decltype(jRange
												                        .max),
												                Integer128<
												                    false,
												                    true>>))
												    {
													    if (maxBigger)
													    {
														    fullIntRange = parse::
														        r129From(
														            fullRange
														                .min,
														            jRange.max);
														    return;
													    }
												    }
												    if constexpr (
												        !(std::same_as<
												              decltype(jRange
												                      .min),
												              Integer128<false,
												                  false>>
												            && std::same_as<
												                decltype(fullRange
												                        .max),
												                Integer128<
												                    false,
												                    true>>))
												    {
													    if (minSmaller)
														    fullIntRange = parse::
														        r129From(
														            jRange.min,
														            fullRange
														                .max);
												    }
											    }
											    return;
										    }
										    // No overlap.
										    throw std::runtime_error(
										        "TODO: error logging, found non overlapping int ranges in type inference");
									    });
								});
							}
						}
						//Make a new type that is a subtype of all the edit
						//types (a variant or just the first element)
						Slu_assert(!types.empty());
						if (types.size() == 1)
							i.resolveNoCheck(types[0]->clone());
						else
						{
							parse::RawTypeKind::Variant v
							    = parse::VariantRawType::newRawTy();
							for (const parse::ResolvedType* j : types)
								v->options.emplace_back(j->clone());
							if (!intRanges.empty())
							{
								parse::r129Get(
								    fullIntRange, [&](const auto& fullRange) {
									    v->options.emplace_back(
									        parse::ResolvedType::newIntRange(
									            fullRange));
								    });
							}
							auto [sz, alignData] = v->calcSizeAndAlign();
							parse::ResolvedType res;
							res.base = std::move(v);
							res.size = sz;
							res.alignmentData = alignData;
							i.resolveNoCheck(std::move(res));
						}
					}
				}
				//Check use's

				visitSubTySide(i.use, [&](const parse::ResolvedType& useTy) {
					if (!useTy.isComplete())
						throw std::runtime_error(
						    "TODO: error logging, found incomplete type expr");
					// Check if the resolved type is a subtype of useTy.
					if (!subtypeCheck(mpDb, *i.resolvedType, useTy))
						throw std::runtime_error(
						    "TODO: error logging, found type that is not a subtype of use type");
				});
				i.use.clear();

				i.taken = false;
			}
		}
		parse::ResolvedType& resolveLocal(LocalVarInfo& local)
		{
			if (local.resolved)
				return *local.resolvedType;

			checkLocals(std::span{&local, 1});
			return *local.resolvedType;
		}

		void postLocals(parse::Locals<Cfg>& itm)
		{
			//Check all restrictions.

			checkLocals(tmpLocalsDataStack.back());
			checkLocals(localsDataStack.back());

			localsStack.pop_back();
			localsDataStack.pop_back();
			tmpLocalsDataStack.pop_back();
			if (!breakPointStack.empty())
				breakPointStack.pop_back();
		}
	};

	export void typeInferAndCheck(
	    const parse::BasicMpDbData& mpDbData, std::span<parse::Stat> module)
	{
		TypeInfCheckVisitor vi{{}, mpDbData};

		for (auto& i : module)
			visit::visitStat(vi, i);
		Slu_assert(vi.exprTypeStack.empty());
	}
} //namespace slu::mlvl
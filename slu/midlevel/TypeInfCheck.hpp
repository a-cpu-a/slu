/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <slu/ext/CppMatch.hpp>
#include <slu/lang/BasicState.hpp>
#include <slu/parser/State.hpp>
#include <slu/parser/BuildState.hpp>
#include <slu/parser/OpTraits.hpp>
#include <slu/visit/Visit.hpp>
#include <slu/midlevel/ResolveType.hpp>

namespace slu::mlvl
{
	using TypeInfCheckCfg = decltype(parse::sluCommon);

	template <class T>
	inline bool sameCheck(const T& itm, const parse::ResolvedType& useTy)
	{
		if (!std::holds_alternative<T>(useTy.base))
			return false;
		const T& o = std::get<T>(useTy.base);
		return itm == o;
	}
	inline bool rangeRangeSubtypeCheck(const parse::AnyRawRange auto itm, const parse::AnyRawRange auto useTy) {
		return itm.max <= useTy.max && itm.min >= useTy.min;
	}
	template <parse::AnyRawIntOrRange T>
	inline bool intRangeSubtypeCheck(const T itm, const parse::ResolvedType& useTy)
	{
		return ezmatch(useTy.base)(
		varcase(const auto&) { return false; },
		varcase(const parse::AnyRawIntOrRange auto) {
			constexpr bool isInt = parse::AnyRawInt<std::remove_cvref_t<decltype(var)>>;
			if constexpr (parse::AnyRawInt<T> && isInt)
				return itm == var;
			else if constexpr (parse::AnyRawInt<T>)
				return var.isInside(itm);
			else if constexpr (isInt)
				return itm.isOnly(var);
			else
				return rangeRangeSubtypeCheck(itm,var);
		}
		);
	}

	inline bool subtypeCheck(parse::BasicMpDb mpDb,const parse::ResolvedType& subty, const parse::ResolvedType& useTy)
	{
		if (subty.outerSliceDims != useTy.outerSliceDims)
			return false;
		if (subty.outerSliceDims != 0)
		{
			//TODO: Slice <= Slice.
		}
		ezmatch(subty.base)(
		varcase(const parse::RawTypeKind::TypeError) {
				return true;//poisioned, so pass forward.
		},
		varcase(const parse::RawTypeKind::String&) {
			return sameCheck(var,useTy);
		},
		varcase(const parse::RawTypeKind::Float64) {
			return sameCheck(var, useTy);//TODO: allow f64 as the type too (needs to be impl first).
		},



		varcase(const parse::AnyRawIntOrRange auto) {
			return intRangeSubtypeCheck(var, useTy);
		}
		);
	}


	//TODO: fix string expressions being fully stolen.
	//TODO: create for func call results, table indexing.
	using TmpVar = uint64_t;

	using VisitTypeBuilder = std::variant<parse::ResolvedType,const parse::ResolvedType*, parse::LocalId, TmpVar>;

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


	struct LocalVarInfo
	{
		std::vector<lang::LocalObjId> fields;
		//traits?
		//???

		SubTySide use;//Requirements when used.
		SubTySide edit;//Requirements when writen to.

		bool boolLike :1 = false;//part of use.

		bool taken : 1 = false;

		bool resolved : 1 = false;

		constexpr parse::ResolvedType* resolvedType()
		{
			if (!resolved)return nullptr;
			return &edit.tys[0];
		}
		parse::ResolvedType& resolveNoCheck(parse::ResolvedType&& t)
		{
			_ASSERT(!resolved);
			//use.clear();
			fields.clear();
			edit.clear();
			resolved = true;
			return edit.tys.emplace_back(std::move(t));
		}
		void requireBoolLike(parse::BasicMpDb mpDb)
		{
			if(resolved && !boolLike && !resolvedType()->isBool(mpDb))
				throw std::runtime_error("TODO: error logging, found non bool expr");

			boolLike = true;
		}
	};
	using LocalVarList = std::vector<LocalVarInfo>;

	struct TypeInfCheckVisitor : visit::EmptyVisitor<TypeInfCheckCfg>
	{
		using Cfg = TypeInfCheckCfg;
		static constexpr bool isSlu = Cfg::settings() & ::slu::parse::sluSyn;

		parse::BasicMpDb mpDb;
		std::vector<parse::Locals<Cfg>*> localsStack;
		std::vector<LocalVarList> localsDataStack;
		std::vector<LocalVarList> tmpLocalsDataStack;

		std::vector<VisitTypeBuilder> exprTypeStack;

		LocalVarInfo& localVar(const parse::LocalId id) {
			return localsDataStack.back()[id.v];
		}
		LocalVarInfo& localVar(const TmpVar id) {
			return tmpLocalsDataStack.back()[id];
		}

		void requireAsBool(const VisitTypeBuilder& t)
		{
			ezmatch(t)(
			varcase(const parse::ResolvedType&) {
				if (!var.isBool(mpDb))
					throw std::runtime_error("TODO: error logging, found non bool expr");
			},
			varcase(const parse::ResolvedType*) {
				if (!var->isBool(mpDb))
					throw std::runtime_error("TODO: error logging, found non bool expr");
			},
			varcase(const parse::LocalId) {
				localVar(var).requireBoolLike(mpDb);
			},
			varcase(const TmpVar) {
				localVar(var).requireBoolLike(mpDb);
			}
			);
		}
		void requireUseTy(const VisitTypeBuilder& t,const parse::ResolvedType& ty)
		{
			ezmatch(t)(
			varcase(const parse::ResolvedType&) {
				if(!subtypeCheck(mpDb,var, ty))
					throw std::runtime_error("TODO: error logging, found non matching type expr");
			},
			varcase(const parse::ResolvedType*) {
				if (!subtypeCheck(mpDb, *var, ty))
					throw std::runtime_error("TODO: error logging, found non matching type expr");
			},
			varcase(const parse::LocalId) {
				localVar(var).use.tyRefs.push_back(&ty);
			},
			varcase(const TmpVar) {
				localVar(var).use.tyRefs.push_back(&ty);
			}
			);
		}

		bool preExpr(parse::Expr<Cfg>& itm) 
		{
			exprTypeStack.emplace_back();
			return false;
		}

		template<class RawT>
		bool handleConstType(auto&& v)
		{
			exprTypeStack.emplace_back(parse::ResolvedType::getConstType(RawT{ std::move(v) }));
			return false;
		}
		void editLocalVar(parse::LocalId itm)
		{
			VisitTypeBuilder& editTy = exprTypeStack.back();
			ezmatch(editTy)(
			varcase(const auto&) {},
			varcase(const parse::LocalId) {
				localVar(var).use.locals.push_back(itm);
			},
			varcase(const TmpVar) {
				localVar(var).use.locals.push_back(itm);
			}
			);
			ezmatch(editTy)(
			varcase(parse::ResolvedType&) {
				localVar(itm).edit.tys.emplace_back(std::move(var));
			},
			varcase(const parse::ResolvedType*) {
				localVar(itm).edit.tyRefs.emplace_back(var);
			},
			varcase(const parse::LocalId) {
				localVar(itm).edit.locals.emplace_back(var);
			},
			varcase(const TmpVar) {
				localVar(itm).edit.tmpLocals.emplace_back(var);
			}
			);
			exprTypeStack.pop_back();
		}

		bool preF64(parse::ExprType::F64 itm) {
			return handleConstType<parse::RawTypeKind::Float64>(itm);
		}
		bool preI64(parse::ExprType::I64 itm) {
			return handleConstType<parse::RawTypeKind::Int64>(itm);
		}
		bool preU64(parse::ExprType::U64 itm) {
			return handleConstType<parse::RawTypeKind::Uint64>(itm);
		}
		bool preI128(parse::ExprType::I128 itm) {
			return handleConstType<parse::RawTypeKind::Int128>(itm);
		}
		bool preU128(parse::ExprType::U128 itm) {
			return handleConstType<parse::RawTypeKind::Uint128>(itm);
		}
		bool preExprString(parse::ExprType::String& itm) {
			return handleConstType<parse::RawTypeKind::String>(std::move(itm.v));//Steal it as converter will use the type anyway.
		}

		//Restrictions.
		void postAnyCond(parse::Expr<Cfg>& itm) {
			requireAsBool(exprTypeStack.back());
			exprTypeStack.pop_back();
		}
		void postCanonicLocal(parse::StatementType::CanonicLocal& itm) {
			editLocalVar(itm.name);
		}
		void postFuncCallStat(parse::StatementType::FuncCall<Cfg>& itm) {
			if(itm.argChain.size() != 1)
				throw std::runtime_error("TODO: type inference for complex func call args.");
			if(!std::holds_alternative<parse::ArgsType::ExprList<Cfg>>(itm.argChain[0].args))
				throw std::runtime_error("TODO: type inference for func call with complex args.");

			if(!std::holds_alternative<parse::LimPrefixExprType::VAR<Cfg>>(*itm.val))
				throw std::runtime_error("TODO: type inference for func call on expr.");
			parse::Var<Cfg>& funcVar = std::get<parse::LimPrefixExprType::VAR<Cfg>>(*itm.val).v;
			if (!funcVar.sub.empty())
				throw std::runtime_error("TODO: type inference for sub variables in func-call statement.");
			if (!std::holds_alternative<parse::BaseVarType::NAME<Cfg>>(funcVar.base))
				throw std::runtime_error("TODO: type inference for func call on non-global var.");

			parse::MpItmId<Cfg> funcName = std::get<parse::BaseVarType::NAME<Cfg>>(funcVar.base).v;
			const parse::ItmType::Fn& funcItm = std::get<parse::ItmType::Fn>(mpDb.data->getItm(funcName));

			parse::ArgsType::ExprList<Cfg>& args = std::get<parse::ArgsType::ExprList<Cfg>>(itm.argChain[0].args);
			//Restrict arg exprs to match types in funcItm.
			for (size_t i = args.size(); i > 0; i++)
			{
				const parse::ResolvedType& ty = funcItm.args[i];
				requireUseTy(exprTypeStack.back(), ty);
				exprTypeStack.pop_back();
			}
			//Make temp var for func result, also add editType for it.
			//const TmpVar tmpVar = TmpVar(tmpLocalsDataStack.back().size());
			//tmpLocalsDataStack.back().emplace_back().editTys.emplace_back(&funcItm.ret);
			//
			//exprTypeStack.emplace_back(tmpVar);
		}
		void postAssign(parse::StatementType::Assign<Cfg>& itm)
		{
			size_t count = itm.vars.size();
			for (size_t i = count; i > 0; i--)
			{
				parse::Var<Cfg>& var = itm.vars[i-1];
				if(!var.sub.empty())
					throw std::runtime_error("TODO: type inference for sub variables in assign statement.");

				ezmatch(var.base)(
				varcase(parse::BaseVarType::NAMEv<true>&) {
					throw std::runtime_error("TODO: type check global assign statement.");
				},
				varcase(parse::BaseVarType::ExprV<true>&) {
					throw std::runtime_error("TODO: type inference for expr-var in assign statement.");
				},
				varcase(parse::BaseVarType::Local&) {
					editLocalVar(var);
				},

				varcase(parse::BaseVarType::Root&) {
					throw std::runtime_error("TODO better logging: cant assign to mp root (:>).");
				}
				);
			}
		}

		//Allow any type.
		void postDrop(parse::StatementType::Drop<Cfg>&) {
			exprTypeStack.pop_back();
		}

		//Ignored.
		bool preCanonicGlobal(parse::StatementType::CanonicGlobal&) {
			return true;
		}
		//Stack stuff.
		bool preLocals(parse::Locals<Cfg>& itm)
		{
			localsStack.push_back(&itm);
			localsDataStack.emplace_back();
			tmpLocalsDataStack.emplace_back();
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

		void checkLocals(std::span<LocalVarInfo> locals)
		{
			for (LocalVarInfo& i : locals)
			{
				if (i.resolved)
					continue;
				if(i.taken)
					throw std::runtime_error("TODO: error logging, variable type depends on itself, cant inferr it");
				i.taken = true;
				// Resolve its type

				if (i.boolLike)
				{// "true", "false" or "bool"
					bool canTrue = false;
					bool canFalse = false;
					bool poison = false;

					//First resolve those things & also check types a bit.

					visitSubTySide(i.edit,
						[&](const parse::ResolvedType& otherTy) {
							if(std::holds_alternative<parse::RawTypeKind::TypeError>(otherTy.base))
							{
								poison = true;
								return;
							}

							if (!otherTy.isComplete())
								throw std::runtime_error("TODO: error logging, found incomplete type expr");
							if(otherTy.size>1)
								throw std::runtime_error("TODO: error logging, found non bool expr");
							auto tyNameOpt = otherTy.getStructName();
							if (!tyNameOpt)
								throw std::runtime_error("TODO: error logging, found non bool expr");
							parse::MpItmIdV<true> tyName = *tyNameOpt;
							if (tyName == mpDb.data->getItm({ "std","bool" }))
								canTrue = canFalse = true;
							else if (tyName == mpDb.data->getItm({ "std","bool", "true" }))
								canTrue = true;
							else if (tyName == mpDb.data->getItm({ "std","bool", "false" }))
								canFalse = true;
							else
								throw std::runtime_error("TODO: error logging, found non bool expr");
						}
					);

					if(!(canTrue || canFalse))
					{
						poison = true;
						//TODO: error logging, found non bool type
					}

					if (poison)
						i.resolveNoCheck(parse::ResolvedType::newError());
					else
						i.resolveNoCheck(parse::ResolvedType::getBool(mpDb,canTrue, canFalse));
				}
				else
				{//TODO: resolve it (find a type, that is a subtype for all the edit types).
					visitSubTySide(i.edit,
						[&](const parse::ResolvedType& editTy) {
							if (!editTy.isComplete())
								throw std::runtime_error("TODO: error logging, found incomplete type expr");
						});
				}
				//Check use's

				visitSubTySide(i.use,
					[&](const parse::ResolvedType& useTy) {
						if (!useTy.isComplete())
							throw std::runtime_error("TODO: error logging, found incomplete type expr");
						// Check if the resolved type is a subtype of useTy.
						if (!subtypeCheck(mpDb, *i.resolvedType(), useTy))
							throw std::runtime_error("TODO: error logging, found type that is not a subtype of use type");
					}
				);
				i.use.clear();

				i.taken = false;
			}
		}
		parse::ResolvedType& resolveLocal(LocalVarInfo& local)
		{
			if (local.resolved)
				return *local.resolvedType();
			
			checkLocals(std::span<LocalVarInfo>{ &local,1 });
			return *local.resolvedType();
		}

		void postLocals(parse::Locals<Cfg>& itm) {
			//Check all restrictions.

			checkLocals(tmpLocalsDataStack.back());
			checkLocals(localsDataStack.back());

			localsStack.pop_back();
			localsDataStack.pop_back();
			tmpLocalsDataStack.pop_back();
		}
	};

	inline void typeInferrAndCheck(parse::BasicMpDbData& mpDbData, lang::MpItmIdV<true> modName, parse::StatListV<true>& module)
	{
		TypeInfCheckVisitor vi{ {},parse::BasicMpDb{ &mpDbData } };

		for (auto& i : module)
			visit::visitStat(vi, i);
		_ASSERT(vi.exprTypeStack.empty());
	}
}
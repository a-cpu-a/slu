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


	//TODO: fix string expressions being fully stolen.
	//TODO: create for func call results, table indexing.
	using TmpVar = uint64_t;

	using VisitTypeBuilder = std::variant<parse::ResolvedType,const parse::ResolvedType*, parse::LocalId, TmpVar>;

	struct TypeRestriction
	{
		//fields?
		//methods?
		//traits?
		//???
	};
	using TypeRestrictions = std::vector<TypeRestriction>;

	struct LocalVarInfo
	{
		std::vector<VisitTypeBuilder> editTys;
		std::vector<TypeRestrictions> useTys;
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
				localsDataStack.back()[var.v].useTys.emplace_back(/*TODO*/);
			},
			varcase(const TmpVar) {
				tmpLocalsDataStack.back()[var].useTys.emplace_back(/*TODO*/);
			}
			);
		}
		void requireAsTy(const VisitTypeBuilder& t,const parse::ResolvedType& ty)
		{
			ezmatch(t)(
			varcase(const parse::ResolvedType&) {
				//TODO: check equivelance / sub type.
			},
			varcase(const parse::ResolvedType*) {
				//TODO: check equivelance / sub type.
			},
			varcase(const parse::LocalId) {
				localsDataStack.back()[var.v].useTys.emplace_back(/*TODO*/);
			},
			varcase(const TmpVar) {
				tmpLocalsDataStack.back()[var].useTys.emplace_back(/*TODO*/);
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
		void editLocalVar(parse::LocalId var)
		{
			localsDataStack.back()[var.v].editTys.emplace_back(exprTypeStack.back());
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
				requireAsTy(exprTypeStack.back(), ty);
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
		void postLocals(parse::Locals<Cfg>& itm) {
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
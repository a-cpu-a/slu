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

	using VisitTypeBuilder = std::variant<parse::ResolvedType, parse::LocalId>;

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
		std::vector<parse::ResolvedType> editTys;
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

		std::vector<VisitTypeBuilder> exprTypeStack;

		void requireAsBool(const VisitTypeBuilder& t)
		{
			ezmatch(t)(
			varcase(const parse::ResolvedType&) {
				if (!var.isBool(mpDb))
					throw std::runtime_error("TODO: error logging, found non bool expr");
			},
			varcase(const parse::LocalId&) {
				localsDataStack.back()[var.v].useTys.emplace_back(/*TODO*/);
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
			exprTypeStack.back().resolved = parse::ResolvedType::getConstType(RawT{ std::move(v)});
			return false;
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
		void postIfCond(parse::Expr<Cfg>& itm) {
			requireAsBool(exprTypeStack.back());
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
			return false;
		}
		void postLocals(parse::Locals<Cfg>& itm) {
			localsStack.pop_back();
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
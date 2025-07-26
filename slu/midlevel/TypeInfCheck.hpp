/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <slu/lang/BasicState.hpp>
#include <slu/parser/State.hpp>
#include <slu/parser/BuildState.hpp>
#include <slu/parser/OpTraits.hpp>
#include <slu/visit/Visit.hpp>
#include <slu/midlevel/ResolveType.hpp>

namespace slu::mlvl
{
	using TypeInfCheckCfg = decltype(parse::sluCommon);

	struct VisitTypeBuilder
	{
		parse::ResolvedType resolved;
		//???
	};
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
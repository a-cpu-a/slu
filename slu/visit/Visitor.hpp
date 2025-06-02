/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <format>
#include <vector>

#include <slu/parser/State.hpp>
#include <slu/Settings.hpp>

namespace slu::visit
{
	template<parse::AnySettings _SettingsT = parse::Setting<void>>
	struct EmptyVisitor
	{
		using SettingsT = _SettingsT;
		using Cfg = parse::VecInput<SettingsT>;//TODO: swap with dummy settings cfg holder
		constexpr static SettingsT settings() { return SettingsT(); }

		constexpr EmptyVisitor(SettingsT) {}
		constexpr EmptyVisitor() = default;

		bool preFile(parse::ParsedFile<Cfg>& itm)
		{
			return false;
		}
		void postFile(parse::ParsedFile<Cfg>& itm)
		{}

		bool preBlock(parse::Block<Cfg>& itm)
		{
			return false;
		}
		void postBlock(parse::Block<Cfg>& itm)
		{}

		bool preVar(parse::Var<Cfg>& itm)
		{
			return false;
		}
		void postVar(parse::Var<Cfg>& itm)
		{}

		bool prePat(parse::Pat<Cfg>& itm)
		{
			return false;
		}
		void postPat(parse::Pat<Cfg>& itm)
		{}

		bool preDestrSpec(parse::DestrSpec<Cfg>& itm)
		{
			return false;
		}
		void postDestrSpec(parse::DestrSpec<Cfg>& itm)
		{}

		bool preSoe(parse::Soe<Cfg>& itm)
		{
			return false;
		}
		void postSoe(parse::Soe<Cfg>& itm)
		{}

		bool preExpr(parse::Expression<Cfg>& itm)
		{
			return false;
		}
		void postExpr(parse::Expression<Cfg>& itm)
		{}

		bool preTypeExp(parse::TypeExpr& itm)
		{
			return false;
		}
		void postTypeExp(parse::TypeExpr& itm)
		{}

		bool preTable(parse::TableConstructor<Cfg>& itm)
		{
			return false;
		}
		void postTable(parse::TableConstructor<Cfg>& itm)
		{}

		bool preStat(parse::Statement<Cfg>& itm)
		{
			return false;
		}
		void postStat(parse::Statement<Cfg>& itm)
		{}

		bool preLimPrefixExpr(parse::LimPrefixExpr<Cfg>& itm)
		{
			return false;
		}
		void postLimPrefixExpr(parse::LimPrefixExpr<Cfg>& itm)
		{}

		bool preDestrSimple(parse::PatType::Simple<Cfg>& itm)
		{
			return false;
		}
		void postDestrSimple(parse::PatType::Simple<Cfg>& itm)
		{}

		bool preBaseVarExpr(parse::BaseVarType::EXPR<Cfg>&itm)
		{
			return false;
		}
		void postBaseVarExpr(parse::BaseVarType::EXPR<Cfg>& itm)
		{}

		bool preBaseVarName(parse::BaseVarType::NAME<Cfg>&itm)
		{
			return false;
		}
		void postBaseVarName(parse::BaseVarType::NAME<Cfg>& itm)
		{}

		//Edge cases:
		bool preDestrField(parse::DestrField<Cfg>& itm)
		{
			return false;
		}
		bool preDestrFieldPat(parse::DestrField<Cfg>& itm)
		{
			return false;
		}
		void postDestrField(parse::DestrField<Cfg>& itm)
		{}

		bool preDestrName(parse::PatType::DestrName<Cfg>& itm)
		{
			return false;
		}
		bool preDestrNameName(parse::PatType::DestrName<Cfg>& itm)
		{
			return false;
		}
		void postDestrName(parse::PatType::DestrName<Cfg>& itm)
		{}

		bool preDestrNameRestrict(parse::PatType::DestrNameRestrict<Cfg>& itm)
		{
			return false;
		}
		bool preDestrNameRestrictName(parse::PatType::DestrNameRestrict<Cfg>& itm)
		{
			return false;
		}
		bool preDestrNameRestriction(parse::PatType::DestrNameRestrict<Cfg>& itm)
		{
			return false;
		}
		void postDestrNameRestrict(parse::PatType::DestrNameRestrict<Cfg>& itm)
		{}

		//Pre only:
		bool preBaseVarRoot(const parse::BaseVarType::Root itm)
		{
			return false;
		}
		bool preBlockReturn(parse::Block<Cfg>& itm)
		{
			return false;
		}
		bool preDestrAny(const parse::PatType::DestrAny itm)
		{
			return false;
		}
		bool preName(parse::MpItmId<Cfg>& itm)
		{
			return false;
		}
		bool preString(std::span<char>&itm)
		{
			return false;
		}

		//Lists:
		bool preDestrList(parse::PatType::DestrList<Cfg>& itm)
		{
			return false;
		}
		bool preDestrListFirst(parse::PatType::DestrList<Cfg>& itm)
		{
			return false;
		}
		void sepDestrList(std::span<parse::Pat<Cfg>> list, parse::Pat<Cfg>& itm)
		{}
		bool preDestrListName(parse::PatType::DestrList<Cfg>&itm)
		{
			return false;
		}
		void postDestrList(parse::PatType::DestrList<Cfg>& itm)
		{}
		
		bool preDestrFields(parse::PatType::DestrFields<Cfg>& itm)
		{
			return false;
		}
		bool preDestrFieldsFirst(parse::PatType::DestrFields<Cfg>& itm)
		{
			return false;
		}
		void sepDestrFields(std::span<parse::DestrField<Cfg>> list, parse::DestrField<Cfg>& itm)
		{}
		bool preDestrFieldsName(parse::PatType::DestrFields<Cfg>&itm)
		{
			return false;
		}
		void postDestrFields(parse::PatType::DestrFields<Cfg>& itm)
		{}
		
		bool preVarList(std::span<parse::Var<Cfg>> itm)
		{
			return false;
		}
		void sepVarList(std::span<parse::Var<Cfg>> list, parse::Var<Cfg>& itm)
		{}
		void postVarList(std::span<parse::Var<Cfg>> itm)
		{}
		
		bool preArgChain(std::span<parse::ArgFuncCall<Cfg>> itm)
		{
			return false;
		}
		void sepArgChain(std::span<parse::ArgFuncCall<Cfg>> list, parse::ArgFuncCall<Cfg>& itm)
		{}
		void postArgChain(std::span<parse::ArgFuncCall<Cfg>> itm)
		{}

		bool preExpList(std::span<parse::Expression<Cfg>> itm)
		{
			return false;
		}
		void sepExpList(std::span<parse::Expression<Cfg>> list, parse::Expression<Cfg>& itm)
		{}
		void postExpList(std::span<parse::Expression<Cfg>> itm)
		{}

		bool preNameList(std::span<parse::MpItmId<Cfg>> itm)
		{
			return false;
		}
		void sepNameList(std::span<parse::MpItmId<Cfg>> list, parse::MpItmId<Cfg>& itm)
		{}
		void postNameList(std::span<parse::MpItmId<Cfg>> itm)
		{}

		bool preParams(std::span<parse::Parameter<Cfg>> itm)
		{
			return false;
		}
		void sepParams(std::span<parse::Parameter<Cfg>> list, parse::Parameter<Cfg>& itm)
		{}
		void postParams(std::span<parse::Parameter<Cfg>> itm)
		{}
	};

	/*
		(bool"shouldStop" pre_) post_
		sep_ -> commas ::'s etc
	*/
	template<class T>
	concept AnyVisitor = parse::AnyCfgable<T> && std::is_base_of_v<EmptyVisitor<typename T::SettingsT>, T>;
}
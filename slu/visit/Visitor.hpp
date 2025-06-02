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

#define _Slu_DEF_EMPTY_POST(_Name,_Ty) void post##_Name(_Ty& itm){}
#define _Slu_DEF_EMPTY_POST_UNIT(_Name,_Ty) void post##_Name(const _Ty itm){}

#define _Slu_DEF_EMPTY_PRE(_Name,_Ty) bool pre##_Name(_Ty& itm){return false;}
#define _Slu_DEF_EMPTY_PRE_UNIT(_Name,_Ty) bool pre##_Name(const _Ty itm){return false;}

#define _Slu_DEF_EMPTY_PRE_POST(_Name,_Ty) _Slu_DEF_EMPTY_PRE(_Name,_Ty); _Slu_DEF_EMPTY_POST(_Name,_Ty);
#define _Slu_DEF_EMPTY_AUTO(_Name)  _Slu_DEF_EMPTY_PRE_POST(_Name,parse:: _Name <Cfg>)

		_Slu_DEF_EMPTY_PRE_POST(File, parse::ParsedFile<Cfg>);
		_Slu_DEF_EMPTY_AUTO(Block);
		_Slu_DEF_EMPTY_AUTO(Var);
		_Slu_DEF_EMPTY_AUTO(Pat);
		_Slu_DEF_EMPTY_AUTO(DestrSpec);
		_Slu_DEF_EMPTY_AUTO(Soe);
		_Slu_DEF_EMPTY_AUTO(LimPrefixExpr);
		_Slu_DEF_EMPTY_PRE_POST(Expr, parse::Expression<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(TypeExp, parse::TypeExpr);
		_Slu_DEF_EMPTY_PRE_POST(Table, parse::TableConstructor<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(Stat, parse::Statement<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(DestrSimple, parse::PatType::Simple<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(BaseVarExpr, parse::BaseVarType::EXPR<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(BaseVarName, parse::BaseVarType::NAME<Cfg>);

		//Edge cases:
		_Slu_DEF_EMPTY_AUTO(DestrField);
		_Slu_DEF_EMPTY_PRE(DestrFieldPat, parse::DestrField<Cfg>);

		_Slu_DEF_EMPTY_PRE_POST(DestrName, parse::PatType::DestrName<Cfg>);
		_Slu_DEF_EMPTY_PRE(DestrNameName, parse::PatType::DestrName<Cfg>);

		_Slu_DEF_EMPTY_PRE_POST(DestrNameRestrict, parse::PatType::DestrNameRestrict<Cfg>);
		_Slu_DEF_EMPTY_PRE(DestrNameRestrictName, parse::PatType::DestrNameRestrict<Cfg>);
		_Slu_DEF_EMPTY_PRE(DestrNameRestriction, parse::PatType::DestrNameRestrict<Cfg>);

		//Pre only:
		_Slu_DEF_EMPTY_PRE_UNIT(BaseVarRoot, parse::BaseVarType::Root);
		_Slu_DEF_EMPTY_PRE_UNIT(DestrAny, parse::PatType::DestrAny);
		_Slu_DEF_EMPTY_PRE(BlockReturn, parse::Block<Cfg>);
		_Slu_DEF_EMPTY_PRE(Name, parse::MpItmId<Cfg>);
		_Slu_DEF_EMPTY_PRE_UNIT(String, std::span<char>);
		_Slu_DEF_EMPTY_PRE_UNIT(True, parse::ExprType::TRUE);
		_Slu_DEF_EMPTY_PRE_UNIT(False, parse::ExprType::FALSE);
		_Slu_DEF_EMPTY_PRE_UNIT(Nil, parse::ExprType::NIL);

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
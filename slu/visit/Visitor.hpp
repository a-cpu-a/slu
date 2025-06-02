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

#define _Slu_DEF_EMPTY_POST_RAW(_Name,_Ty) void post##_Name(_Ty itm){}
#define _Slu_DEF_EMPTY_POST(_Name,_Ty) _Slu_DEF_EMPTY_POST_RAW(_Name,_Ty&)
#define _Slu_DEF_EMPTY_POST_UNIT(_Name,_Ty) _Slu_DEF_EMPTY_POST_RAW(_Name,const _Ty)

#define _Slu_DEF_EMPTY_PRE_RAW(_Name,_Ty) bool pre##_Name(_Ty itm){return false;}
#define _Slu_DEF_EMPTY_PRE(_Name,_Ty) _Slu_DEF_EMPTY_PRE_RAW(_Name,_Ty&)
#define _Slu_DEF_EMPTY_PRE_UNIT(_Name,_Ty) _Slu_DEF_EMPTY_PRE_RAW(_Name,const _Ty)

#define _Slu_DEF_EMPTY_PRE_POST(_Name,_Ty) _Slu_DEF_EMPTY_PRE(_Name,_Ty); _Slu_DEF_EMPTY_POST(_Name,_Ty);
#define _Slu_DEF_EMPTY_AUTO(_Name)  _Slu_DEF_EMPTY_PRE_POST(_Name,parse:: _Name <Cfg>)

#define _Slu_DEF_EMPTY_SEP_RAW(_Name,_Ty,_ElemTy) void sep##_Name(_Ty list,_ElemTy itm){}
#define _Slu_DEF_EMPTY_SEP(_Name,_Ty,_ElemTy) _Slu_DEF_EMPTY_SEP_RAW(_Name,_Ty,_ElemTy&)

#define _Slu_DEF_EMPTY_LIST(_Name,_ElemTy) \
	_Slu_DEF_EMPTY_PRE_RAW(_Name,std::span<_ElemTy>) \
	_Slu_DEF_EMPTY_POST_RAW(_Name,std::span<_ElemTy>) \
	_Slu_DEF_EMPTY_SEP(_Name,std::span<_ElemTy>,_ElemTy)

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
		_Slu_DEF_EMPTY_PRE(DestrList, parse::PatType::DestrList<Cfg>);
		_Slu_DEF_EMPTY_PRE(DestrListFirst, parse::PatType::DestrList<Cfg>);
		_Slu_DEF_EMPTY_PRE(DestrListName, parse::PatType::DestrList<Cfg>);
		_Slu_DEF_EMPTY_POST(DestrList, parse::PatType::DestrList<Cfg>);
		_Slu_DEF_EMPTY_SEP(DestrList, std::span<parse::Pat<Cfg>>, parse::Pat<Cfg>);
		
		_Slu_DEF_EMPTY_PRE(DestrFields, parse::PatType::DestrFields<Cfg>);
		_Slu_DEF_EMPTY_PRE(DestrFieldsFirst, parse::PatType::DestrFields<Cfg>);
		_Slu_DEF_EMPTY_PRE(DestrFieldsName, parse::PatType::DestrFields<Cfg>);
		_Slu_DEF_EMPTY_POST(DestrFields, parse::PatType::DestrFields<Cfg>);
		_Slu_DEF_EMPTY_SEP(DestrFields, std::span<parse::DestrField<Cfg>>, parse::DestrField<Cfg>);
		
		_Slu_DEF_EMPTY_LIST(VarList, parse::Var<Cfg>);
		_Slu_DEF_EMPTY_LIST(ArgChain, parse::ArgFuncCall<Cfg>);
		_Slu_DEF_EMPTY_LIST(ExpList, parse::Expression<Cfg>);
		_Slu_DEF_EMPTY_LIST(NameList, parse::MpItmId<Cfg>);
		_Slu_DEF_EMPTY_LIST(Params, parse::Parameter<Cfg>);
	};

	/*
		(bool"shouldStop" pre_) post_
		sep_ -> commas ::'s etc
	*/
	template<class T>
	concept AnyVisitor = parse::AnyCfgable<T> && std::is_base_of_v<EmptyVisitor<typename T::SettingsT>, T>;
}
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

#define Slu_co ,

namespace slu::visit
{
#define Slu_esc(...) __VA_ARGS__

	template<parse::AnySettings _SettingsT = parse::Setting<void>>
	struct EmptyVisitor
	{
		using SettingsT = _SettingsT;
		using Cfg = parse::VecInput<SettingsT>;//TODO: swap with dummy settings cfg holder
		constexpr static SettingsT settings() { return SettingsT(); }

		constexpr EmptyVisitor(SettingsT) {}
		constexpr EmptyVisitor() = default;

#define _Slu_DEF_EMPTY_POST_RAW(_Name,...) void post##_Name(__VA_ARGS__ itm){}
#define _Slu_DEF_EMPTY_POST_RAW_LG(_Name,_TY_POST_SYMBOL,...) \
	_Slu_DEF_EMPTY_POST_RAW(_Name##Local,__VA_ARGS__<Cfg,true> _TY_POST_SYMBOL);\
	_Slu_DEF_EMPTY_POST_RAW(_Name##Global,__VA_ARGS__<Cfg,false> _TY_POST_SYMBOL)
#define _Slu_DEF_EMPTY_POST(_Name,_Ty) _Slu_DEF_EMPTY_POST_RAW(_Name,_Ty&)
#define _Slu_DEF_EMPTY_POST_LG(_Name,_Ty) _Slu_DEF_EMPTY_POST_RAW_LG(_Name,&,_Ty)
#define _Slu_DEF_EMPTY_POST_UNIT(_Name,_Ty) _Slu_DEF_EMPTY_POST_RAW(_Name,const _Ty)

#define _Slu_DEF_EMPTY_PRE_RAW(_Name,...) bool pre##_Name(__VA_ARGS__ itm){return false;}
#define _Slu_DEF_EMPTY_PRE_RAW_LG(_Name,_TY_POST_SYMBOL,...) \
	_Slu_DEF_EMPTY_PRE_RAW(_Name##Local,__VA_ARGS__<Cfg,true> _TY_POST_SYMBOL);\
	_Slu_DEF_EMPTY_PRE_RAW(_Name##Global,__VA_ARGS__<Cfg,false> _TY_POST_SYMBOL)

#define _Slu_DEF_EMPTY_PRE(_Name,...) _Slu_DEF_EMPTY_PRE_RAW(_Name,__VA_ARGS__&)
#define _Slu_DEF_EMPTY_PRE_LG(_Name,...) _Slu_DEF_EMPTY_PRE_RAW_LG(_Name,&,__VA_ARGS__)
#define _Slu_DEF_EMPTY_PRE_UNIT(_Name,...) _Slu_DEF_EMPTY_PRE_RAW(_Name,const __VA_ARGS__)

#define _Slu_DEF_EMPTY_PRE_POST_RAW(_Name,...) \
	_Slu_DEF_EMPTY_PRE_RAW(_Name,__VA_ARGS__); \
	_Slu_DEF_EMPTY_POST_RAW(_Name,__VA_ARGS__)
#define _Slu_DEF_EMPTY_PRE_POST_RAW_LG(_Name,_TY_POST_SYMBOL,...) \
	_Slu_DEF_EMPTY_PRE_RAW_LG(_Name,_TY_POST_SYMBOL,__VA_ARGS__); \
	_Slu_DEF_EMPTY_POST_RAW_LG(_Name,_TY_POST_SYMBOL,__VA_ARGS__)
#define _Slu_DEF_EMPTY_PRE_POST(_Name,...) _Slu_DEF_EMPTY_PRE_POST_RAW(_Name,__VA_ARGS__&)
#define _Slu_DEF_EMPTY_PRE_POST_LG(_Name,...) _Slu_DEF_EMPTY_PRE_POST_RAW_LG(_Name,&,__VA_ARGS__)
#define _Slu_DEF_EMPTY_PRE_POST_UNIT(_Name,...) _Slu_DEF_EMPTY_PRE_POST_RAW(_Name,const __VA_ARGS__)

#define _Slu_DEF_EMPTY_AUTO(_Name)  _Slu_DEF_EMPTY_PRE_POST(_Name,parse:: _Name <Cfg>)
#define _Slu_DEF_EMPTY_AUTO_LG(_Name)  _Slu_DEF_EMPTY_PRE_POST_LG(_Name,parse:: _Name)

#define _Slu_DEF_EMPTY_SEP_RAW(_Name,_Ty,_ElemTy) void sep##_Name(_Ty list,_ElemTy itm){}
#define _Slu_DEF_EMPTY_SEP(_Name,_Ty,_ElemTy) _Slu_DEF_EMPTY_SEP_RAW(_Name,Slu_esc(_Ty),Slu_esc(_ElemTy&))

#define _Slu_DEF_EMPTY_LIST(_Name,_ElemTy) \
	_Slu_DEF_EMPTY_PRE_RAW(_Name,std::span<_ElemTy>) \
	_Slu_DEF_EMPTY_POST_RAW(_Name,std::span<_ElemTy>) \
	_Slu_DEF_EMPTY_SEP(_Name,std::span<_ElemTy>,_ElemTy)

#define _Slu_DEF_EMPTY_AUTO(_Name)  _Slu_DEF_EMPTY_PRE_POST(_Name,parse:: _Name <Cfg>)


#define _Slu_DEF_EMPTY_LIST(_Name,_ElemTy) \
	_Slu_DEF_EMPTY_PRE_RAW(_Name,std::span<_ElemTy>) \
	_Slu_DEF_EMPTY_POST_RAW(_Name,std::span<_ElemTy>) \
	_Slu_DEF_EMPTY_SEP(_Name,std::span<_ElemTy>,_ElemTy)

		_Slu_DEF_EMPTY_PRE_POST(File, parse::ParsedFile<Cfg>);
		_Slu_DEF_EMPTY_AUTO(Block);
		_Slu_DEF_EMPTY_AUTO(Var);
		_Slu_DEF_EMPTY_AUTO_LG(Pat);
		_Slu_DEF_EMPTY_AUTO(DestrSpec);
		_Slu_DEF_EMPTY_AUTO(Soe);
		_Slu_DEF_EMPTY_AUTO(LimPrefixExpr);
		_Slu_DEF_EMPTY_AUTO(FunctionInfo);
		_Slu_DEF_EMPTY_PRE_POST(Lifetime, parse::Lifetime);
		_Slu_DEF_EMPTY_PRE_POST(Expr, parse::Expression<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(ExprString, parse::ExprType::LITERAL_STRING);
		_Slu_DEF_EMPTY_PRE_POST(MultiOp, parse::ExprType::MULTI_OPERATION<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(TypeMultiOp, parse::TypeExprDataType::MULTI_OP);
		_Slu_DEF_EMPTY_PRE_POST(TraitExpr, parse::TraitExpr);
		_Slu_DEF_EMPTY_PRE_POST(Table, parse::TableConstructor<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(Stat, parse::Statement<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(DestrSimple, parse::PatType::Simple<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(BaseVarExpr, parse::BaseVarType::EXPR<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(BaseVarName, parse::BaseVarType::NAME<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(ExternBlock, parse::StatementType::ExternBlock<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(StatList, parse::StatList<Cfg>);

		//Edge cases:
		_Slu_DEF_EMPTY_PRE_POST(TypeExpr, parse::TypeExpr);
		_Slu_DEF_EMPTY_PRE(TypeExprMut, parse::TypeExpr);

		_Slu_DEF_EMPTY_AUTO_LG(DestrField);
		_Slu_DEF_EMPTY_PRE_LG(DestrFieldPat, parse::DestrField);

		_Slu_DEF_EMPTY_PRE_POST_LG(DestrName, parse::PatType::DestrName);
		_Slu_DEF_EMPTY_PRE_LG(DestrNameName, parse::PatType::DestrName);

		_Slu_DEF_EMPTY_PRE_POST_LG(DestrNameRestrict, parse::PatType::DestrNameRestrict);
		_Slu_DEF_EMPTY_PRE_LG(DestrNameRestrictName, parse::PatType::DestrNameRestrict);
		_Slu_DEF_EMPTY_PRE_LG(DestrNameRestriction, parse::PatType::DestrNameRestrict);

		_Slu_DEF_EMPTY_PRE_POST(UnOp, parse::UnOpItem);
		_Slu_DEF_EMPTY_PRE(UnOpMut, parse::UnOpItem);
		_Slu_DEF_EMPTY_PRE(UnOpConst, parse::UnOpItem);
		_Slu_DEF_EMPTY_PRE(UnOpShare, parse::UnOpItem);
		//Pre only:
		_Slu_DEF_EMPTY_PRE_UNIT(PostUnOp, parse::PostUnOpType);
		_Slu_DEF_EMPTY_PRE_UNIT(BinOp, parse::BinOpType);
		_Slu_DEF_EMPTY_PRE_UNIT(BaseVarRoot, parse::BaseVarType::Root);
		_Slu_DEF_EMPTY_PRE_UNIT(DestrAny, parse::PatType::DestrAny);
		_Slu_DEF_EMPTY_PRE(BlockReturn, parse::Block<Cfg>);
		_Slu_DEF_EMPTY_PRE(Name, parse::MpItmId<Cfg>);
		_Slu_DEF_EMPTY_PRE_UNIT(String, std::span<char>);
		_Slu_DEF_EMPTY_PRE_UNIT(True, parse::ExprType::TRUE);
		_Slu_DEF_EMPTY_PRE_UNIT(False, parse::ExprType::FALSE);
		_Slu_DEF_EMPTY_PRE_UNIT(Nil, parse::ExprType::NIL);

		//Lists:
		_Slu_DEF_EMPTY_PRE_LG(DestrList, parse::PatType::DestrList);
		_Slu_DEF_EMPTY_PRE_LG(DestrListFirst, parse::PatType::DestrList);
		_Slu_DEF_EMPTY_PRE_LG(DestrListName, parse::PatType::DestrList);
		_Slu_DEF_EMPTY_POST_LG(DestrList, parse::PatType::DestrList);
		_Slu_DEF_EMPTY_SEP(DestrListLocal, std::span<parse::Pat<Cfg Slu_co true>>, parse::Pat<Cfg Slu_co true>);
		_Slu_DEF_EMPTY_SEP(DestrListGlobal, std::span<parse::Pat<Cfg Slu_co false>>, parse::Pat<Cfg Slu_co false>);

		_Slu_DEF_EMPTY_PRE_LG(DestrFields, parse::PatType::DestrFields);
		_Slu_DEF_EMPTY_PRE_LG(DestrFieldsFirst, parse::PatType::DestrFields);
		_Slu_DEF_EMPTY_PRE_LG(DestrFieldsName, parse::PatType::DestrFields);
		_Slu_DEF_EMPTY_POST_LG(DestrFields, parse::PatType::DestrFields);
		_Slu_DEF_EMPTY_SEP(DestrFieldsLocal, std::span<parse::DestrField<Cfg Slu_co true>>, parse::DestrField<Cfg Slu_co true>);
		_Slu_DEF_EMPTY_SEP(DestrFieldsGlobal, std::span<parse::DestrField<Cfg Slu_co false>>, parse::DestrField<Cfg Slu_co false>);

		_Slu_DEF_EMPTY_LIST(VarList, parse::Var<Cfg>);
		_Slu_DEF_EMPTY_LIST(ArgChain, parse::ArgFuncCall<Cfg>);
		_Slu_DEF_EMPTY_LIST(ExprList, parse::Expression<Cfg>);
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
module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <string>
#include <span>
#include <format>
#include <vector>

export module slu.visit.empty;

import slu.settings;
import slu.ast.enums;
import slu.ast.state;
import slu.ast.state_decls;

#define Slu_co ,

namespace slu::visit
{
#define Slu_esc(...) __VA_ARGS__

	export template<parse::AnySettings _SettingsT = parse::Setting<void>>
	struct EmptyVisitor
	{
		using SettingsT = _SettingsT;
		using Cfg = SettingsT;
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

		_Slu_DEF_EMPTY_PRE_POST(File, parse::ParsedFile);
		_Slu_DEF_EMPTY_AUTO(Block);
		_Slu_DEF_EMPTY_AUTO_LG(Pat);
		_Slu_DEF_EMPTY_AUTO(Soe);
		_Slu_DEF_EMPTY_PRE_POST(DestrSpec,parse::DestrSpec);
		_Slu_DEF_EMPTY_PRE_POST(FunctionInfo,parse::FunctionInfo);
		_Slu_DEF_EMPTY_PRE_POST(Lifetime, parse::Lifetime);
		_Slu_DEF_EMPTY_PRE_POST(Expr, parse::Expr);
		_Slu_DEF_EMPTY_PRE_POST(TypeExpr, parse::Expr);
		_Slu_DEF_EMPTY_PRE_POST(TableExpr, parse::ExprType::Table<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(GlobalExpr, parse::ExprType::Global<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(DerefExpr, parse::ExprType::Deref);
		_Slu_DEF_EMPTY_PRE_POST(IndexExpr, parse::ExprType::Index);
		_Slu_DEF_EMPTY_PRE_POST(FieldExpr, parse::ExprType::Field<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(CallExpr, parse::ExprType::Call);
		_Slu_DEF_EMPTY_PRE_POST(SelfCallExpr, parse::ExprType::SelfCall);
		_Slu_DEF_EMPTY_PRE_POST(TraitExpr, parse::TraitExpr);
		_Slu_DEF_EMPTY_PRE_POST(ExprString, parse::ExprType::String);
		_Slu_DEF_EMPTY_PRE_POST(MultiOp, parse::ExprType::MultiOp<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(Table, parse::Table<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(Locals, parse::Locals<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(Stat, parse::Stat);
		_Slu_DEF_EMPTY_PRE_POST(DestrSimple, parse::PatType::Simple<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(LocalVar, parse::StatType::Local<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(LetVar, parse::StatType::Let<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(ConstVar, parse::StatType::Const<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(Trait, parse::StatType::Trait);
		_Slu_DEF_EMPTY_PRE_POST(Impl, parse::StatType::Impl);
		_Slu_DEF_EMPTY_PRE_POST(Use, parse::StatType::Use);
		_Slu_DEF_EMPTY_PRE_POST(AnyCond, parse::Expr);
		_Slu_DEF_EMPTY_PRE_POST(IfExpr, parse::ExprType::IfCond<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(IfStat, parse::StatType::IfCond<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(ExternBlock, parse::StatType::ExternBlock<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(AnyFuncDefStat, parse::StatType::Function);
		_Slu_DEF_EMPTY_PRE_POST(AnyFuncDeclStat, parse::StatType::FunctionDecl<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(Assign, parse::StatType::Assign<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(CanonicLocal, parse::StatType::CanonicLocal);
		_Slu_DEF_EMPTY_PRE_POST(CanonicGlobal, parse::StatType::CanonicGlobal);
		_Slu_DEF_EMPTY_PRE_POST(Drop, parse::StatType::Drop<Cfg>);
		_Slu_DEF_EMPTY_PRE_POST(CallStat, parse::StatType::Call);
		_Slu_DEF_EMPTY_PRE_POST(SelfCallStat, parse::StatType::SelfCall);
		_Slu_DEF_EMPTY_PRE_POST(StatList, parse::StatList<Cfg>);

		//Edge cases:
		_Slu_DEF_EMPTY_PRE_POST_RAW(LocalExpr, parse::ExprType::Local);
		_Slu_DEF_EMPTY_PRE_POST_RAW(OpenRange, parse::ExprType::OpenRange);
		_Slu_DEF_EMPTY_PRE_POST_RAW(F64, parse::ExprType::F64);
		_Slu_DEF_EMPTY_PRE_POST_RAW(I64, parse::ExprType::I64);
		_Slu_DEF_EMPTY_PRE_POST_RAW(U64, parse::ExprType::U64);
		_Slu_DEF_EMPTY_PRE_POST_RAW(P128, parse::ExprType::P128);
		_Slu_DEF_EMPTY_PRE_RAW(M128, parse::ExprType::M128);
		_Slu_DEF_EMPTY_PRE_RAW(Infer, parse::ExprType::Infer);

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
		_Slu_DEF_EMPTY_PRE_UNIT(PostUnOp, ast::PostUnOpType);
		_Slu_DEF_EMPTY_PRE_UNIT(BinOp, ast::BinOpType);
		_Slu_DEF_EMPTY_PRE_UNIT(DestrAny, parse::PatType::DestrAny<Cfg Slu_co true>);
		_Slu_DEF_EMPTY_PRE_UNIT(DestrAny, parse::PatType::DestrAny<Cfg Slu_co false>);
		_Slu_DEF_EMPTY_PRE(BlockReturn, parse::Block<Cfg>);
		_Slu_DEF_EMPTY_PRE(Name, lang::MpItmId);
		_Slu_DEF_EMPTY_PRE_UNIT(String, std::span<char>);
		_Slu_DEF_EMPTY_PRE_UNIT(True, parse::ExprType::True);
		_Slu_DEF_EMPTY_PRE_UNIT(False, parse::ExprType::False);
		_Slu_DEF_EMPTY_PRE_UNIT(Nil, parse::ExprType::Nil);

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

		_Slu_DEF_EMPTY_LIST(ExprList, parse::Expr);
		_Slu_DEF_EMPTY_LIST(NameList, lang::MpItmId);
		_Slu_DEF_EMPTY_LIST(Params, parse::Parameter);
	};


	/*
		(bool"shouldStop" pre_) post_
		sep_ -> commas ::'s etc
	*/
	export template<class T>
	concept AnyVisitor = parse::AnyCfgable<T> && std::is_base_of_v<EmptyVisitor<typename T::SettingsT>, T>;
}
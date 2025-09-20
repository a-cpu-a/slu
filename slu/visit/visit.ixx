module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <string>
#include <span>
#include <format>
#include <vector>

#include <slu/Panic.hpp>
#include <slu/ext/CppMatch.hpp>
export module slu.visit.visit;

import slu.settings;
import slu.ast.enums;
import slu.ast.state;
import slu.ast.state_decls;
import slu.lang.basic_state;
import slu.visit.empty;

namespace slu::visit
{
#define Slu_CALL_VISIT_FN_PRE_USER(_Name,_itm) do {\
		if(vi.pre ## _Name (_itm)) \
			return;} while (false)
#define Slu_CALL_VISIT_FN_PRE_USER_LG(_Name,_itm) \
		if constexpr (isLocal) \
			{Slu_CALL_VISIT_FN_PRE_USER(_Name##Local,_itm);} \
		else \
			{Slu_CALL_VISIT_FN_PRE_USER(_Name##Global,_itm);}
#define Slu_CALL_VISIT_FN_PRE(_Name) Slu_CALL_VISIT_FN_PRE_USER(_Name,itm)
#define Slu_CALL_VISIT_FN_PRE_VAR(_Name) Slu_CALL_VISIT_FN_PRE_USER(_Name,var)
#define Slu_CALL_VISIT_FN_PRE_LG(_Name) Slu_CALL_VISIT_FN_PRE_USER_LG(_Name,itm)
#define Slu_CALL_VISIT_FN_PRE_VAR_LG(_Name) Slu_CALL_VISIT_FN_PRE_USER_LG(_Name,var)

#define Slu_CALL_VISIT_FN_SEP(_Name,_i,_vec) do {\
		if(&_i != &_vec.back()) \
			vi.sep##_Name(_vec,_i); } while (false)
#define Slu_CALL_VISIT_FN_SEP_LG(_Name,_i,_itm) \
		if constexpr (isLocal) \
			{Slu_CALL_VISIT_FN_SEP(_Name##Local,_i,_itm);} \
		else \
			{Slu_CALL_VISIT_FN_SEP(_Name##Global,_i,_itm);}

#define Slu_CALL_VISIT_FN_POST_USER(_Name,_itm) \
		vi.post##_Name(_itm)
#define Slu_CALL_VISIT_FN_POST_USER_LG(_Name,_itm) \
		if constexpr (isLocal) \
			{Slu_CALL_VISIT_FN_POST_USER(_Name##Local,_itm);} \
		else \
			{Slu_CALL_VISIT_FN_POST_USER(_Name##Global,_itm);}
#define Slu_CALL_VISIT_FN_POST(_Name) Slu_CALL_VISIT_FN_POST_USER(_Name,itm)
#define Slu_CALL_VISIT_FN_POST_VAR(_Name) Slu_CALL_VISIT_FN_POST_USER(_Name,var)
#define Slu_CALL_VISIT_FN_POST_LG(_Name) Slu_CALL_VISIT_FN_POST_USER_LG(_Name,itm)
#define Slu_CALL_VISIT_FN_POST_VAR_LG(_Name) Slu_CALL_VISIT_FN_POST_USER_LG(_Name,var)

	export template<AnyVisitor Vi>
	void visitString(Vi& vi, std::span<char> itm)
	{
		Slu_CALL_VISIT_FN_PRE(String);
	}
	export template<AnyVisitor Vi>
	void visitName(Vi& vi, lang::MpItmId& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Name);
	}
	export template<AnyVisitor Vi>
	void visitPoolString(Vi& vi, lang::PoolString& itm)
	{
		//TODO
	}
	export template<bool isLocal,AnyVisitor Vi>
	void visitNameOrLocal(Vi& vi, parse::LocalOrName<Vi,isLocal>& itm)
	{
		if constexpr (isLocal)
		{//TODO
		}
		else
			visitName(vi, itm);
	}
	export template<AnyVisitor Vi>
	void visitMp(Vi& vi, lang::MpItmId& itm)
	{
		//TODO
	}
	export template<AnyVisitor Vi>
	void visitExported(Vi& vi, const lang::ExportData itm)
	{
		//TODO
	}
	export template<AnyVisitor Vi>
	void visitSafety(Vi& vi, const ast::OptSafety itm)
	{
		//TODO
	}
	export template<AnyVisitor Vi>
	void visitNameList(Vi& vi, parse::NameList<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(NameList);
		for (auto& i : itm)
		{
			visitName(vi, i);
			Slu_CALL_VISIT_FN_SEP(NameList, i, itm);
		}
		Slu_CALL_VISIT_FN_POST(NameList);
	}
	export template<AnyVisitor Vi>
	void visitDestrSpec(Vi& vi, parse::DestrSpec& itm)
	{
		Slu_CALL_VISIT_FN_PRE(DestrSpec);
		ezmatch(itm)(
		varcase(parse::DestrSpecType::Spat&) {
			visitExpr(vi, var);
		},
		varcase(parse::DestrSpecType::Prefix&) {
			visitUnOps(vi, var);
		}
		);
		Slu_CALL_VISIT_FN_POST(DestrSpec);
	}
	export template<bool isLocal, AnyVisitor Vi>
	void visitPat(Vi& vi, parse::Pat<Vi,isLocal>& itm)
	{
		Slu_CALL_VISIT_FN_PRE_LG(Pat);
		ezmatch(itm)(
		varcase(const parse::PatType::DestrAny<Vi,isLocal>) {
			Slu_CALL_VISIT_FN_PRE_VAR(DestrAny);
			//TODO: synthetic name visit?
		},

		varcase(parse::PatType::Simple<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(DestrSimple);
			visitExpr(vi, var);
			Slu_CALL_VISIT_FN_POST_VAR(DestrSimple);
		},

		varcase(parse::PatType::DestrFields<Vi,isLocal>&) {
			Slu_CALL_VISIT_FN_PRE_VAR_LG(DestrFields);
			visitDestrSpec(vi, var.spec);
			Slu_CALL_VISIT_FN_PRE_VAR_LG(DestrFieldsFirst);
			for (auto& i : var.items)
			{
				Slu_CALL_VISIT_FN_PRE_USER_LG(DestrField, i);
				visitPoolString(vi, i.name);
				Slu_CALL_VISIT_FN_PRE_USER_LG(DestrFieldPat, i);
				visitPat<isLocal>(vi, i.pat);
				Slu_CALL_VISIT_FN_POST_USER_LG(DestrField, i);
				Slu_CALL_VISIT_FN_SEP_LG(DestrFields, i, var.items);
			}
			Slu_CALL_VISIT_FN_PRE_VAR_LG(DestrFieldsName);
			visitNameOrLocal<isLocal>(vi, var.name);
			Slu_CALL_VISIT_FN_POST_VAR_LG(DestrFields);
		},
		varcase(parse::PatType::DestrList<Vi, isLocal>&) {
			Slu_CALL_VISIT_FN_PRE_VAR_LG(DestrList);
			visitDestrSpec(vi, var.spec);
			Slu_CALL_VISIT_FN_PRE_VAR_LG(DestrListFirst);
			for (auto& i : var.items)
			{
				visitPat<isLocal>(vi, i);
				Slu_CALL_VISIT_FN_SEP_LG(DestrList, i, var.items);
			}
			Slu_CALL_VISIT_FN_PRE_VAR_LG(DestrListName);
			visitNameOrLocal<isLocal>(vi, var.name);
			Slu_CALL_VISIT_FN_POST_VAR_LG(DestrList);
		},

		varcase(parse::PatType::DestrName<Vi, isLocal>&) {
			Slu_CALL_VISIT_FN_PRE_VAR_LG(DestrName);
			visitDestrSpec(vi, var.spec);
			if (!var.name.empty())
			{
				Slu_CALL_VISIT_FN_PRE_VAR_LG(DestrNameName);
				visitNameOrLocal<isLocal>(vi, var.name);
			}
			Slu_CALL_VISIT_FN_POST_VAR_LG(DestrName);
		},
		varcase(parse::PatType::DestrNameRestrict<Vi, isLocal>&) {
			Slu_CALL_VISIT_FN_PRE_VAR_LG(DestrNameRestrict);
			visitDestrSpec(vi, var.spec);
			if (!var.name.empty())
			{
				Slu_CALL_VISIT_FN_PRE_VAR_LG(DestrNameRestrictName);
				visitNameOrLocal<isLocal>(vi, var.name);
			}
			Slu_CALL_VISIT_FN_PRE_VAR_LG(DestrNameRestriction);
			visitExpr(vi, var.restriction);
			Slu_CALL_VISIT_FN_POST_VAR_LG(DestrNameRestrict);
		}
		);
		Slu_CALL_VISIT_FN_POST_LG(Pat);
	}
	export template<AnyVisitor Vi>
	void visitStatList(Vi& vi, parse::StatList<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(StatList);
		for (auto& i : itm)
			visitStat(vi, i);
		Slu_CALL_VISIT_FN_POST(StatList);
	}
	export template<AnyVisitor Vi>
	void visitTypeExpr(Vi& vi, parse::Expr& itm)
	{
		Slu_CALL_VISIT_FN_PRE(TypeExpr);
		visitExpr(vi, itm);
		Slu_CALL_VISIT_FN_POST(TypeExpr);
	}
	export template<AnyVisitor Vi>
	void visitTraitExpr(Vi& vi, parse::TraitExpr& itm)
	{
		Slu_CALL_VISIT_FN_PRE(TraitExpr);
		//TODO
		Slu_CALL_VISIT_FN_POST(TraitExpr);
	}
	export template<AnyVisitor Vi>
	void visitLifetime(Vi& vi, parse::Lifetime& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Lifetime);
		//TODO
		Slu_CALL_VISIT_FN_POST(Lifetime);
	}
	export template<AnyVisitor Vi>
	void visitBinOp(Vi& vi, const ast::BinOpType itm) {
		Slu_CALL_VISIT_FN_PRE(BinOp);
	}
	export template<AnyVisitor Vi>
	void visitUnOps(Vi& vi, std::span<parse::UnOpItem> list) 
	{
		for (auto& itm :list)
		{
			Slu_CALL_VISIT_FN_PRE(UnOp);
			if (itm.type == ast::UnOpType::REF
				|| itm.type == ast::UnOpType::REF_MUT
				|| itm.type == ast::UnOpType::REF_CONST
				|| itm.type == ast::UnOpType::REF_SHARE)
			{
				visitLifetime(vi, itm.life);
				if (itm.type == ast::UnOpType::REF_MUT)
					Slu_CALL_VISIT_FN_PRE(UnOpMut);
				else if (itm.type == ast::UnOpType::REF_CONST)
					Slu_CALL_VISIT_FN_PRE(UnOpConst);
				else if (itm.type == ast::UnOpType::REF_SHARE)
					Slu_CALL_VISIT_FN_PRE(UnOpShare);
			}
			Slu_CALL_VISIT_FN_POST(UnOp);
		}
	}
	export template<AnyVisitor Vi>
	void visitPostUnOps(Vi& vi, std::span<const ast::PostUnOpType> list) 
	{
		for (auto& itm : list) {
			Slu_CALL_VISIT_FN_PRE(PostUnOp);
		}
	}
	export template<AnyVisitor Vi>
	void visitExpr(Vi& vi, parse::Expr& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Expr);
		visitUnOps(vi, itm.unOps);
		visitExprData(vi, itm.data);
		visitPostUnOps(vi, std::span<const ast::PostUnOpType>{ itm.postUnOps.data(), itm.postUnOps.size()});
		Slu_CALL_VISIT_FN_POST(Expr);
	}
	export template<AnyVisitor Vi>
	void visitExprData(Vi& vi, parse::ExprData<Vi>& itm)
	{
		ezmatch(itm)(
		varcase(const parse::ExprType::False) {
			Slu_CALL_VISIT_FN_PRE_VAR(False);
		},
		varcase(const parse::ExprType::True) {
			Slu_CALL_VISIT_FN_PRE_VAR(True);
		},
		varcase(const parse::ExprType::Nil) {
			Slu_CALL_VISIT_FN_PRE_VAR(Nil);
		},
		varcase(parse::ExprType::String&) {
			Slu_CALL_VISIT_FN_PRE_VAR(ExprString);
			visitString(vi, var.v);
			Slu_CALL_VISIT_FN_POST_VAR(ExprString);
		},
		varcase(const parse::ExprType::F64) {
			Slu_CALL_VISIT_FN_PRE_VAR(F64);
			//TODO
		},
		varcase(const parse::ExprType::I64) {
			Slu_CALL_VISIT_FN_PRE_VAR(I64);
			//TODO
		},
		varcase(const parse::ExprType::P128) {
			Slu_CALL_VISIT_FN_PRE_VAR(P128);
			//TODO
		},
		varcase(const parse::ExprType::U64) {
			Slu_CALL_VISIT_FN_PRE_VAR(U64);
			//TODO
		},
		varcase(const parse::ExprType::M128) {
			Slu_CALL_VISIT_FN_PRE_VAR(M128);
			//TODO
		},
		varcase(const parse::ExprType::OpenRange) {
			Slu_CALL_VISIT_FN_PRE_VAR(OpenRange);
		},
		varcase(const parse::ExprType::VarArgs) {
			//TODO
		},
		varcase(parse::ExprType::Parens<Vi>&) {
			visitExpr(vi, *var);
		},
		varcase(const parse::ExprType::MpRoot) {
			//TODO
		},
		varcase(const parse::ExprType::Local) {
			Slu_CALL_VISIT_FN_PRE_VAR(LocalExpr);
			//TODO
			Slu_CALL_VISIT_FN_POST_VAR(LocalExpr);
		},
		varcase(parse::ExprType::Global<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(GlobalExpr);
			visitMp(vi, var);
			Slu_CALL_VISIT_FN_POST_VAR(GlobalExpr);
		},
		varcase(parse::ExprType::Deref&) {
			Slu_CALL_VISIT_FN_PRE_VAR(DerefExpr);
			visitExpr(vi, *var.v);
			Slu_CALL_VISIT_FN_POST_VAR(DerefExpr);
		},
		varcase(parse::ExprType::Index&) {
			Slu_CALL_VISIT_FN_PRE_VAR(IndexExpr);
			visitExpr(vi, *var.v);
			visitExpr(vi, *var.idx);
			Slu_CALL_VISIT_FN_POST_VAR(IndexExpr);
		},
		varcase(parse::ExprType::Field<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(FieldExpr);
			visitExpr(vi, *var.v);
			visitPoolString(vi, var.field);
			Slu_CALL_VISIT_FN_POST_VAR(FieldExpr);
		},
		varcase(parse::ExprType::Call&) {
			Slu_CALL_VISIT_FN_PRE_VAR(CallExpr);
			visitExpr(vi, *var.v);
			visitArgs(vi, var.args);
			Slu_CALL_VISIT_FN_POST_VAR(CallExpr);
		},
		varcase(parse::ExprType::SelfCall&) {
			Slu_CALL_VISIT_FN_PRE_VAR(SelfCallExpr);
			visitExpr(vi, *var.v);
			visitName(vi, var.method);
			visitArgs(vi, var.args);
			Slu_CALL_VISIT_FN_POST_VAR(SelfCallExpr);
		},
		varcase(parse::ExprType::TraitExpr&) {
			//TODO: pre post
			visitTraitExpr(vi, var);
		},
		varcase(parse::ExprType::IfCond<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(IfExpr);
			Slu_CALL_VISIT_FN_PRE_USER(AnyCond, *var.cond);
			visitExpr(vi, *var.cond);
			Slu_CALL_VISIT_FN_POST_USER(AnyCond, *var.cond);
			visitSoe(vi, *var.bl);
			for (auto& [cond, soe] : var.elseIfs)
			{
				Slu_CALL_VISIT_FN_PRE_USER(AnyCond, cond);
				visitExpr(vi, cond);
				Slu_CALL_VISIT_FN_POST_USER(AnyCond, cond);
				visitSoe(vi, soe);
			}
			if (var.elseBlock.has_value())
				visitSoe(vi, **var.elseBlock);
			Slu_CALL_VISIT_FN_POST_VAR(IfExpr);
		},
		varcase(parse::ExprType::Lifetime&) {
			//TODO: pre post
			visitLifetime(vi,var);
		},
		varcase(parse::ExprType::Table<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(TableExpr);
			visitTable(vi,var);
			Slu_CALL_VISIT_FN_POST_VAR(TableExpr);
		},
		varcase(parse::ExprType::Function&) {
			//TODO: pre post
			Slu_CALL_VISIT_FN_PRE_VAR(FunctionInfo);
			visitSafety(vi, var.safety);
			Slu_CALL_VISIT_FN_PRE_USER(Locals, var.local2Mp);
			visitParams(vi, var.params);
			if(var.retType.has_value())
				visitTypeExpr(vi, **var.retType);
			Slu_CALL_VISIT_FN_POST_USER(Locals, var.local2Mp);
			Slu_CALL_VISIT_FN_POST_VAR(FunctionInfo);
			visitBlock(vi, var.block);
		},
		varcase(parse::ExprType::MultiOp<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(MultiOp);
			visitExpr(vi, *var.first);
			for (auto& [op,expr] : var.extra)
			{
				visitBinOp(vi, op);
				visitExpr(vi, expr);
			}
			Slu_CALL_VISIT_FN_POST_VAR(MultiOp);
		},
		varcase(parse::ExprType::PatTypePrefix&) {
			Slu_panic("PatTypePrefix is not a valid expression, somehow leaked out of parsing!!!");
		},


		varcase(parse::ExprType::Err&) {
			visitTypeExpr(vi, *var.err);
		},
		varcase(const parse::ExprType::Infer) {
			Slu_CALL_VISIT_FN_PRE_VAR(Infer);
		},
		varcase(parse::ExprType::Dyn&) {
			visitTraitExpr(vi, var.expr);
		},
		varcase(parse::ExprType::Impl&) {
			visitTraitExpr(vi, var.expr);
		},
		varcase(parse::ExprType::Struct&) {
			visitTable(vi, var.fields);
		},
		varcase(parse::ExprType::Union&) {
			visitTable(vi, var.fields);
		},
		varcase(parse::ExprType::FnType&) {
			visitSafety(vi, var.safety);
			visitTypeExpr(vi, *var.argType);
			visitTypeExpr(vi, *var.retType);
		}
		);
	}
	export template<AnyVisitor Vi>
	void visitExprList(Vi& vi, parse::ExprList& itm)
	{
		Slu_CALL_VISIT_FN_PRE(ExprList);
		for (auto& i : itm)
		{
			visitExpr(vi, i);
			Slu_CALL_VISIT_FN_SEP(ExprList,i,itm);
		}
		Slu_CALL_VISIT_FN_POST(ExprList);
	}
	export template<AnyVisitor Vi>
	void visitArgs(Vi& vi, parse::Args& itm)
	{
		ezmatch(itm)(
		varcase(parse::ArgsType::ExprList&) { visitExprList(vi, var); },
		varcase(parse::ArgsType::Table<Vi>&) {	  visitTable(vi, var); },
		varcase(parse::ArgsType::String&) {		  visitString(vi,var.v); }
		);
	}
	export template<AnyVisitor Vi>
	void visitTable(Vi& vi, parse::Table<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Table);
		for (auto& i : itm)
		{
			ezmatch(i)(
			varcase(parse::FieldType::NONE&) {},
			varcase(parse::FieldType::Expr&) {
				visitExpr(vi, var);
			},
			varcase(parse::FieldType::Expr2Expr&) {
				visitExpr(vi, var.idx); visitExpr(vi, var.v);
			},
			varcase(parse::FieldType::Name2Expr&) {
				visitExpr(vi, var.v);
			}
			);
		}
		Slu_CALL_VISIT_FN_POST(Table);
	}
	//TODO: var-args
	export template<bool isLocal, AnyVisitor Vi>
	inline void visitParams(Vi& vi, parse::ParamList<isLocal>& itm)
	{
		if constexpr (isLocal)
			Slu_CALL_VISIT_FN_PRE(Params);
		else
			Slu_CALL_VISIT_FN_PRE(ConstParams);
		for (parse::Parameter<isLocal>& i : itm)
		{
			visitNameOrLocal<isLocal>(vi, i.name);
			visitTypeExpr(vi, i.type);
			if constexpr (isLocal)
				Slu_CALL_VISIT_FN_SEP(Params, i, itm);
			else
				Slu_CALL_VISIT_FN_SEP(ConstParams, i, itm);
		}
		if constexpr (isLocal)
			Slu_CALL_VISIT_FN_POST(Params);
		else
			Slu_CALL_VISIT_FN_POST(ConstParams);
	}
	export template<AnyVisitor Vi>
	void visitWhereClauses(Vi& vi, parse::WhereClauses& itm)
	{
		for (auto& i : itm)
		{
			visitName(vi, i.var);
			visitTraitExpr(vi, i.bound);
		}
		//TODO
	}

	export template<AnyVisitor Vi>
	void visitStat(Vi& vi, parse::Stat& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Stat);
		ezmatch(itm.data)(
		varcase(parse::StatType::Assign<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(Assign);
			for (auto& i : var.vars)
			{
				visitExprData(vi, i);
			}
			visitExprList(vi, var.exprs);
			Slu_CALL_VISIT_FN_POST_VAR(Assign);
		},
		varcase(parse::StatType::Local<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(LocalVar);
			visitExported(vi, var.exported);
			visitPat<true>(vi, var.names);
			visitExprList(vi, var.exprs);
			Slu_CALL_VISIT_FN_POST_VAR(LocalVar);
		},
		varcase(parse::StatType::CanonicLocal&) {
			Slu_CALL_VISIT_FN_PRE_VAR(CanonicLocal);
			visitExported(vi, var.exported);
			//visitTypeExpr(vi, var.type); //TODO: resolved type visit?
			visitNameOrLocal<true>(vi, var.name);
			visitExpr(vi, var.value);
			Slu_CALL_VISIT_FN_POST_VAR(CanonicLocal);
		},
		varcase(parse::StatType::CanonicGlobal&) {
			Slu_CALL_VISIT_FN_PRE_VAR(CanonicGlobal);
			visitExported(vi, var.exported);
			//visitTypeExpr(vi, var.type); //TODO: resolved type visit?
			visitName(vi, var.name);
			Slu_CALL_VISIT_FN_PRE_USER(Locals, var.local2Mp);
			visitExpr(vi, var.value);
			Slu_CALL_VISIT_FN_POST_USER(Locals, var.local2Mp);
			Slu_CALL_VISIT_FN_POST_VAR(CanonicGlobal);
		},
		varcase(parse::StatType::Let<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(LetVar);
			visitExported(vi, var.exported);
			visitPat<true>(vi, var.names);
			visitExprList(vi, var.exprs);
			Slu_CALL_VISIT_FN_POST_VAR(LetVar);
		},
		varcase(parse::StatType::Const<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(ConstVar);
			visitExported(vi, var.exported);
			visitPat<false>(vi, var.names);
			Slu_CALL_VISIT_FN_PRE_USER(Locals, var.local2Mp);
			visitExprList(vi, var.exprs);
			Slu_CALL_VISIT_FN_POST_USER(Locals, var.local2Mp);
			Slu_CALL_VISIT_FN_POST_VAR(ConstVar);
		},
		varcase(parse::StatType::Call&) {
			visitExpr(vi, *var.v);
			visitArgs(vi, var.args);
			//TODO
		},
		varcase(parse::StatType::SelfCall&) {
			visitExpr(vi, *var.v);
			visitName(vi, var.method);
			visitArgs(vi, var.args);
			//TODO
		},
		varcase(parse::StatType::Block<Vi>&) {
			visitBlock(vi, var);
			//TODO
		},
		varcase(parse::StatType::Goto<Vi>&) {
			//TODO
		},
		varcase(parse::StatType::Label<Vi>&) {
			//TODO
		},
		varcase(parse::StatType::Use&) {
			Slu_CALL_VISIT_FN_PRE_VAR(Use);
			visitExported(vi, var.exported);
			visitMp(vi, var.base);
			//TODO var.useVariant
			Slu_CALL_VISIT_FN_POST_VAR(Use);
		},
		varcase(parse::StatType::Semicol&) {
			//TODO
		},
		varcase(parse::StatType::IfCond<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(IfStat);
			Slu_CALL_VISIT_FN_PRE_USER(AnyCond, *var.cond);
			visitExpr(vi, *var.cond);
			Slu_CALL_VISIT_FN_POST_USER(AnyCond,*var.cond);
			visitSoe(vi, *var.bl);
			for (auto& [cond, soe] : var.elseIfs)
			{
				Slu_CALL_VISIT_FN_PRE_USER(AnyCond, cond);
				visitExpr(vi, cond);
				Slu_CALL_VISIT_FN_POST_USER(AnyCond, cond);
				visitSoe(vi, soe);
			}
			if (var.elseBlock.has_value())
				visitSoe(vi, **var.elseBlock);
			Slu_CALL_VISIT_FN_POST_VAR(IfStat);
		},
		varcase(parse::StatType::While<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_USER(AnyCond, var.cond);
			visitExpr(vi, var.cond);
			Slu_CALL_VISIT_FN_POST_USER(AnyCond, var.cond);
			visitBlock(vi, var.bl);
		},
		varcase(parse::StatType::RepeatUntil<Vi>&) {
			visitBlock(vi, var.bl);
			Slu_CALL_VISIT_FN_PRE_USER(AnyCond, var.cond);
			visitExpr(vi, var.cond);
			Slu_CALL_VISIT_FN_POST_USER(AnyCond, var.cond);
		},
		varcase(parse::StatType::ForIn<Vi>&) {
			visitPat<true>(vi, var.varNames);
			visitExpr(vi, var.exprs);
			visitBlock(vi, var.bl);
		},
		varcase(parse::StatType::Struct&) {
			visitExported(vi, var.exported);
			visitName(vi, var.name);
			Slu_CALL_VISIT_FN_PRE_USER(Locals, var.local2Mp);
			visitParams(vi, var.params);
			visitTable(vi, var.type);
			Slu_CALL_VISIT_FN_POST_USER(Locals, var.local2Mp);
		},
		varcase(parse::StatType::Union&) {
			visitExported(vi, var.exported);
			visitName(vi, var.name);
			Slu_CALL_VISIT_FN_PRE_USER(Locals, var.local2Mp);
			visitParams(vi, var.params);
			visitTable(vi, var.type);
			Slu_CALL_VISIT_FN_POST_USER(Locals, var.local2Mp);
		},
		varcase(parse::StatType::Function&) {
			Slu_CALL_VISIT_FN_PRE_VAR(AnyFuncDefStat);
			Slu_CALL_VISIT_FN_PRE_USER(FunctionInfo,var.func);
			visitExported(vi, var.exported);
			visitSafety(vi, var.func.safety);
			visitName(vi, var.name);
			Slu_CALL_VISIT_FN_PRE_USER(Locals, var.func.local2Mp);
			visitParams(vi, var.func.params);
			if (var.func.retType.has_value())
				visitTypeExpr(vi, **var.func.retType);
			Slu_CALL_VISIT_FN_POST_USER(FunctionInfo, var.func);
			visitBlock(vi, var.func.block);
			Slu_CALL_VISIT_FN_POST_USER(Locals, var.func.local2Mp);
			Slu_CALL_VISIT_FN_POST_VAR(AnyFuncDefStat);
		},
		varcase(parse::StatType::FunctionDecl<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(AnyFuncDeclStat);
			Slu_CALL_VISIT_FN_PRE_VAR(FunctionInfo);
			visitExported(vi, var.exported);
			visitSafety(vi, var.safety);
			visitName(vi, var.name);
			Slu_CALL_VISIT_FN_PRE_USER(Locals, var.local2Mp);
			visitParams(vi, var.params);
			if (var.retType.has_value())
				visitTypeExpr(vi, **var.retType);
			Slu_CALL_VISIT_FN_POST_USER(Locals, var.local2Mp);
			Slu_CALL_VISIT_FN_POST_VAR(FunctionInfo);
			Slu_CALL_VISIT_FN_POST_VAR(AnyFuncDeclStat);
		},
		varcase(parse::StatType::ExternBlock<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(ExternBlock);
			visitSafety(vi, var.safety);
			visitString(vi, var.abi);
			visitStatList(vi, var.stats);
			Slu_CALL_VISIT_FN_POST_VAR(ExternBlock);
		},
		varcase(parse::StatType::UnsafeBlock<Vi>&) {
			visitStatList(vi, var.stats);
		},
		varcase(const parse::StatType::UnsafeLabel) {
			//TODO
		},
		varcase(const parse::StatType::SafeLabel) {
			//TODO
		},
		varcase(parse::StatType::Drop<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(Drop);
			visitExpr(vi, var.expr);
			Slu_CALL_VISIT_FN_POST_VAR(Drop);
		},
		varcase(parse::StatType::Trait&) {
			Slu_CALL_VISIT_FN_PRE_VAR(Trait);
			visitExported(vi, var.exported);
			visitName(vi, var.name);
			visitParams(vi, var.params);
			if (var.whereSelf.has_value())
				visitTraitExpr(vi, *var.whereSelf);
			visitWhereClauses(vi, var.clauses);
			visitStatList(vi, var.itms);
			Slu_CALL_VISIT_FN_POST_VAR(Trait);
		},
		varcase(parse::StatType::Impl&) {
			Slu_CALL_VISIT_FN_PRE_VAR(Impl);
			visitExported(vi, var.exported);
			visitParams(vi, var.params);
			if (var.forTrait.has_value())
				visitTraitExpr(vi, *var.forTrait);
			visitExpr(vi, var.type);
			visitWhereClauses(vi, var.clauses);
			visitStatList(vi, var.code);
			Slu_CALL_VISIT_FN_POST_VAR(Impl);
		},
		varcase(parse::StatType::Mod<Vi>&) {
			visitExported(vi, var.exported);
			visitName(vi, var.name);
			//TODO
		},
		varcase(parse::StatType::ModAs<Vi>&) {
			visitExported(vi, var.exported);
			visitName(vi, var.name);
			visitStatList(vi, var.code);
		}
		);
		Slu_CALL_VISIT_FN_POST(Stat);
	}
	export template<AnyVisitor Vi>
	void visitSoe(Vi& vi, parse::Soe<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Soe);
		ezmatch(itm)(
		varcase(parse::SoeType::Block<Vi>&) {
			visitBlock(vi, var);
		},
		varcase(parse::SoeType::Expr&) {
			visitExpr(vi, *var);
		}
		);
		Slu_CALL_VISIT_FN_POST(Soe);
	}
	export template<AnyVisitor Vi>
	void visitBlock(Vi& vi, parse::Block<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Block);
		visitStatList(vi, itm.statList);
		if (itm.retTy != parse::RetType::NONE)
		{
			Slu_CALL_VISIT_FN_PRE(BlockReturn);
			visitExprList(vi, itm.retExprs);
		}
		Slu_CALL_VISIT_FN_POST(Block);
	}
	export template<AnyVisitor Vi>
	void visitFile(Vi& vi,parse::ParsedFile& itm)
	{
		Slu_CALL_VISIT_FN_PRE(File);
		visitStatList(vi, itm.code);
		Slu_CALL_VISIT_FN_POST(File);
	}
}
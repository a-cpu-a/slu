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
#include <slu/visit/Visitor.hpp>

namespace slu::visit
{
#define Slu_CALL_VISIT_FN_PRE_USER(_Name,_itm) \
		if(vi.pre ## _Name (_itm)) \
			return
#define Slu_CALL_VISIT_FN_PRE(_Name) Slu_CALL_VISIT_FN_PRE_USER(_Name,itm)
#define Slu_CALL_VISIT_FN_PRE_VAR(_Name) Slu_CALL_VISIT_FN_PRE_USER(_Name,var)

#define Slu_CALL_VISIT_FN_SEP(_Name,_i,_vec) \
		if(&_i != &_vec.back()) \
			vi.sep##_Name(_vec,_i)

#define Slu_CALL_VISIT_FN_POST_USER(_Name,_itm) \
		vi.post##_Name(_itm)
#define Slu_CALL_VISIT_FN_POST(_Name) Slu_CALL_VISIT_FN_POST_USER(_Name,itm)
#define Slu_CALL_VISIT_FN_POST_VAR(_Name) Slu_CALL_VISIT_FN_POST_USER(_Name,var)

	template<AnyVisitor Vi>
	inline void visitString(Vi& vi, std::span<char> itm)
	{
		Slu_CALL_VISIT_FN_PRE(String);
	}
	template<AnyVisitor Vi>
	inline void visitName(Vi& vi, parse::MpItmId<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Name);
	}
	template<AnyVisitor Vi>
	inline void visitMp(Vi& vi, parse::MpItmId<Vi>& itm)
	{
		//TODO
	}
	template<AnyVisitor Vi>
	inline void visitExported(Vi& vi, const parse::ExportData itm)
	{
		//TODO
	}
	template<AnyVisitor Vi>
	inline void visitSafety(Vi& vi, const parse::OptSafety itm)
	{
		//TODO
	}
	template<AnyVisitor Vi>
	inline void visitNameList(Vi& vi, parse::NameList<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(NameList);
		for (auto& i : itm)
		{
			visitName(vi, i);
			Slu_CALL_VISIT_FN_SEP(NameList, i, itm);
		}
		Slu_CALL_VISIT_FN_POST(NameList);
	}
	template<AnyVisitor Vi>
	inline void visitDestrSpec(Vi& vi, parse::DestrSpec<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(DestrSpec);
		ezmatch(itm)(
		varcase(parse::DestrSpecType::Spat<Vi>&) {
			visitExpr(vi, var);
		},
		varcase(parse::DestrSpecType::Type&) {
			visitTypeExp(vi, var);
		},
		varcase(parse::DestrSpecType::Prefix&) {
			visitUnOps(vi, var);
		}
		);
		Slu_CALL_VISIT_FN_POST(DestrSpec);
	}
	template<AnyVisitor Vi>
	inline void visitPat(Vi& vi, parse::Pat<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Pat);
		ezmatch(itm)(
		varcase(const parse::PatType::DestrAny) {
			Slu_CALL_VISIT_FN_PRE_VAR(DestrAny);
		},

		varcase(parse::PatType::Simple<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(DestrSimple);
			visitExpr(vi, var);
			Slu_CALL_VISIT_FN_POST_VAR(DestrSimple);
		},

		varcase(parse::PatType::DestrFields<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(DestrFields);
			visitDestrSpec(vi, var.spec);
			Slu_CALL_VISIT_FN_PRE_VAR(DestrFieldsFirst);
			for (auto& i : var.items)
			{
				Slu_CALL_VISIT_FN_PRE_USER(DestrField, i);
				visitName(vi, i.name);
				Slu_CALL_VISIT_FN_PRE_USER(DestrFieldPat, i);
				visitPat(vi, i.pat);
				Slu_CALL_VISIT_FN_POST_USER(DestrField, i);
				Slu_CALL_VISIT_FN_SEP(DestrFields, i, var.items);
			}
			if(!var.name.empty())
			{
				Slu_CALL_VISIT_FN_PRE_VAR(DestrFieldsName);
				visitName(vi, var.name);
			}
			Slu_CALL_VISIT_FN_POST_VAR(DestrFields);
		},
		varcase(parse::PatType::DestrList<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(DestrList);
			visitDestrSpec(vi, var.spec);
			Slu_CALL_VISIT_FN_PRE_VAR(DestrListFirst);
			for (auto& i : var.items)
			{
				visitPat(vi, i);
				Slu_CALL_VISIT_FN_SEP(DestrList, i, var.items);
			}
			if (!var.name.empty())
			{
				Slu_CALL_VISIT_FN_PRE_VAR(DestrListName);
				visitName(vi, var.name);
			}
			Slu_CALL_VISIT_FN_POST_VAR(DestrList);
		},

		varcase(parse::PatType::DestrName<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(DestrName);
			visitDestrSpec(vi, var.spec);
			if (!var.name.empty())
			{
				Slu_CALL_VISIT_FN_PRE_VAR(DestrNameName);
				visitName(vi, var.name);
			}
			Slu_CALL_VISIT_FN_POST_VAR(DestrName);
		},
		varcase(parse::PatType::DestrNameRestrict<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(DestrNameRestrict);
			visitDestrSpec(vi, var.spec);
			if (!var.name.empty())
			{
				Slu_CALL_VISIT_FN_PRE_VAR(DestrNameRestrictName);
				visitName(vi, var.name);
			}
			Slu_CALL_VISIT_FN_PRE_VAR(DestrNameRestriction);
			visitExpr(vi, var.restriction);
			Slu_CALL_VISIT_FN_POST_VAR(DestrNameRestrict);
		}
		);
		Slu_CALL_VISIT_FN_POST(Pat);
	}
	template<AnyVisitor Vi>
	inline void visitVar(Vi& vi, parse::Var<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Var);
		ezmatch(itm.base)(
		varcase(parse::BaseVarType::EXPR<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(BaseVarExpr);
			visitExpr(vi, var.start);
			Slu_CALL_VISIT_FN_POST_VAR(BaseVarExpr);
		},
		varcase(parse::BaseVarType::NAME<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(BaseVarName);
			visitMp(vi, var.v);
			Slu_CALL_VISIT_FN_POST_VAR(BaseVarName);
		},
		varcase(const parse::BaseVarType::Root) {
			Slu_CALL_VISIT_FN_PRE_VAR(BaseVarRoot);
		}
		);
		for (auto& i : itm.sub)
		{
			visitArgChain(vi,i.funcCalls);
			ezmatch(i.idx)(
			varcase(parse::SubVarType::NAME<Vi>&) {},
			varcase(parse::SubVarType::EXPR<Vi>&) {
				visitExpr(vi, var.idx);
			},
			varcase(parse::SubVarType::DEREF) {}
			);
		}
		Slu_CALL_VISIT_FN_POST(Var);
	}
	template<AnyVisitor Vi>
	inline void visitVarList(Vi& vi, std::span<parse::Var<Vi>> itm)
	{
		Slu_CALL_VISIT_FN_PRE(VarList);
		for (auto& i : itm)
		{
			visitVar(vi,i);
			Slu_CALL_VISIT_FN_SEP(VarList,i,itm);
		}
		Slu_CALL_VISIT_FN_POST(VarList);
	}
	template<AnyVisitor Vi>
	inline void visitTypeExp(Vi& vi, parse::TypeExpr& itm)
	{
		Slu_CALL_VISIT_FN_PRE(TypeExp);
		if (itm.hasMut)
		{
			Slu_CALL_VISIT_FN_PRE(TypeExpMut);
		}
		visitUnOps(vi, itm.unOps);
		//TODO
		visitPostUnOps(vi, std::span<const parse::PostUnOpType>{ itm.postUnOps.data(), itm.postUnOps.size()});
		Slu_CALL_VISIT_FN_POST(TypeExp);
	}
	template<AnyVisitor Vi>
	inline void visitTraitExp(Vi& vi, parse::TraitExpr& itm)
	{
		Slu_CALL_VISIT_FN_PRE(TraitExp);
		//TODO
		Slu_CALL_VISIT_FN_POST(TraitExp);
	}
	template<AnyVisitor Vi>
	inline void visitLifetime(Vi& vi, parse::Lifetime& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Lifetime);
		//TODO
		Slu_CALL_VISIT_FN_POST(Lifetime);
	}
	template<AnyVisitor Vi>
	inline void visitBinOp(Vi& vi, const parse::BinOpType itm) {
		Slu_CALL_VISIT_FN_PRE(BinOp);
	}
	template<AnyVisitor Vi>
	inline void visitUnOps(Vi& vi, std::span<parse::UnOpItem> list) 
	{
		for (auto& itm :list)
		{
			Slu_CALL_VISIT_FN_PRE(UnOp);
			if (itm.type == parse::UnOpType::TO_REF
				|| itm.type == parse::UnOpType::TO_REF_MUT
				|| itm.type == parse::UnOpType::TO_REF_CONST
				|| itm.type == parse::UnOpType::TO_REF_SHARE)
			{
				visitLifetime(vi, itm.life);
				if (itm.type == parse::UnOpType::TO_REF_MUT)
					Slu_CALL_VISIT_FN_PRE(UnOpMut);
				else if (itm.type == parse::UnOpType::TO_REF_CONST)
					Slu_CALL_VISIT_FN_PRE(UnOpConst);
				else if (itm.type == parse::UnOpType::TO_REF_SHARE)
					Slu_CALL_VISIT_FN_PRE(UnOpShare);
			}
			Slu_CALL_VISIT_FN_POST(UnOp);
		}
	}
	template<AnyVisitor Vi>
	inline void visitPostUnOps(Vi& vi, std::span<const parse::PostUnOpType> list) 
	{
		for (auto& itm : list) {
			Slu_CALL_VISIT_FN_PRE(PostUnOp);
		}
	}
	template<AnyVisitor Vi>
	inline void visitExpr(Vi& vi, parse::Expression<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Expr);
		visitUnOps(vi, itm.unOps);
		ezmatch(itm.data)(
		varcase(const parse::ExprType::FALSE) {
			Slu_CALL_VISIT_FN_PRE_VAR(False);
		},
		varcase(const parse::ExprType::TRUE) {
			Slu_CALL_VISIT_FN_PRE_VAR(True);
		},
		varcase(const parse::ExprType::NIL) {
			Slu_CALL_VISIT_FN_PRE_VAR(Nil);
		},
		varcase(parse::ExprType::LITERAL_STRING&) {
			Slu_CALL_VISIT_FN_PRE_VAR(ExprString);
		},
		varcase(const parse::ExprType::NUMERAL) {
			//TODO
		},
		varcase(const parse::ExprType::NUMERAL_I64) {
			//TODO
		},
		varcase(const parse::ExprType::NUMERAL_I128) {
			//TODO
		},
		varcase(const parse::ExprType::NUMERAL_U64) {
			//TODO
		},
		varcase(const parse::ExprType::NUMERAL_U128) {
			//TODO
		},
		varcase(const parse::ExprType::OPEN_RANGE) {
			//TODO
		},
		varcase(const parse::ExprType::VARARGS) {
			//TODO
		},
		varcase(parse::ExprType::TYPE_EXPR&) {
			visitTypeExp(vi, var);
		},
		varcase(parse::ExprType::TRAIT_EXPR&) {
			visitTraitExp(vi, var);
		},
		varcase(parse::ExprType::IfCond<Vi>&) {
			visitSoe(vi, *var.bl);
			if (var.elseBlock.has_value())
				visitSoe(vi, **var.elseBlock);
			visitExpr(vi, *var.cond);
			for (auto& [cond, soe] : var.elseIfs)
			{
				visitExpr(vi, cond);
				visitSoe(vi, soe);
			}
		},
		varcase(parse::ExprType::LIM_PREFIX_EXP<Vi>&) {
			visitLimPrefixExpr(vi, *var);
		},
		varcase(parse::ExprType::FUNC_CALL<Vi>&) {
			visitLimPrefixExpr(vi, *var.val);
			visitArgChain(vi, var.argChain);
		},
		varcase(parse::ExprType::LIFETIME&) {
			visitLifetime(vi,var);
		},
		varcase(parse::ExprType::TABLE_CONSTRUCTOR<Vi>&) {
			visitTable(vi,var.v);
		},
		varcase(parse::ExprType::FUNCTION_DEF<Vi>&) {
			visitSafety(vi, var.v.safety);
			visitParams(vi, var.v.params);
			if(var.v.retType.has_value())
				visitTypeExp(vi, *var.v.retType);
			visitBlock(vi, var.v.block);
			//TODO
		},
		varcase(parse::ExprType::MULTI_OPERATION<Vi>&) {
			Slu_CALL_VISIT_FN_PRE_VAR(MultiOp);
			visitExpr(vi, *var.first);
			for (auto& [op,expr] : var.extra)
			{
				visitBinOp(vi, op);
				visitExpr(vi, expr);
			}
			Slu_CALL_VISIT_FN_POST_VAR(MultiOp);
		},
		varcase(parse::ExprType::PAT_TYPE_PREFIX&) {
			Slu_panic("PatTypePrefix is not a valid expression, somehow leaked out of parsing!!!");
		}
		);
		visitPostUnOps(vi, std::span<const parse::PostUnOpType>{ itm.postUnOps.data(), itm.postUnOps.size()});

		Slu_CALL_VISIT_FN_POST(Expr);
	}
	template<AnyVisitor Vi>
	inline void visitExpList(Vi& vi, parse::ExpList<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(ExpList);
		for (auto& i : itm)
		{
			visitExpr(vi, i);
			Slu_CALL_VISIT_FN_SEP(ExpList,i,itm);
		}
		Slu_CALL_VISIT_FN_POST(ExpList);
	}
	template<AnyVisitor Vi>
	inline void visitLimPrefixExpr(Vi& vi, parse::LimPrefixExpr<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(LimPrefixExpr);
		ezmatch(itm)(
		varcase(parse::LimPrefixExprType::EXPR<Vi>&) { visitExpr(vi, var.v); },
		varcase(parse::LimPrefixExprType::VAR<Vi>&) {
			visitVar(vi, var.v);
		}
		);
		Slu_CALL_VISIT_FN_POST(LimPrefixExpr);
	}
	template<AnyVisitor Vi>
	inline void visitArgChain(Vi& vi, std::span<parse::ArgFuncCall<Vi>> itm)
	{
		Slu_CALL_VISIT_FN_PRE(ArgChain);
		for (auto& i : itm)
		{
			ezmatch(i.args)(
			varcase(parse::ArgsType::EXPLIST<Vi>&) { visitExpList(vi, var.v); },
			varcase(parse::ArgsType::TABLE<Vi>&) {
				visitTable(vi, var.v);
			},
			varcase(const parse::ArgsType::LITERAL&) {}
			);
			Slu_CALL_VISIT_FN_SEP(ArgChain,i,itm);
		}
		Slu_CALL_VISIT_FN_POST(ArgChain);
	}
	template<AnyVisitor Vi>
	inline void visitTable(Vi& vi, parse::TableConstructor<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Table);
		for (auto& i : itm)
		{
			ezmatch(i)(
			varcase(parse::FieldType::NONE&) {},
			varcase(parse::FieldType::EXPR<Vi>&) {
				visitExpr(vi, var.v);
			},
			varcase(parse::FieldType::EXPR2EXPR<Vi>&) {
				visitExpr(vi, var.idx); visitExpr(vi, var.v);
			},
			varcase(parse::FieldType::NAME2EXPR<Vi>&) {
				visitExpr(vi, var.v);
			}
			);
		}
		Slu_CALL_VISIT_FN_POST(Table);
	}
	//TODO: var-args
	template<AnyVisitor Vi>
	inline void visitParams(Vi& vi, parse::ParamList<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Params);
		for (auto& i : itm)
		{
			visitPat(vi, i.name);
			Slu_CALL_VISIT_FN_SEP(Params, i, itm);
		}
		Slu_CALL_VISIT_FN_POST(Params);
	}

	template<AnyVisitor Vi>
	inline void visitStat(Vi& vi, parse::Statement<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Stat);
		ezmatch(itm.data)(
		varcase(parse::StatementType::ASSIGN<Vi>&) {
			visitExpList(vi, var.exprs);
			visitVarList(vi, var.vars);
		},
		varcase(parse::StatementType::LOCAL_ASSIGN<Vi>&) {
			visitExported(vi, var.exported);
			visitPat(vi, var.names);
			visitExpList(vi, var.exprs);
		},
		varcase(parse::StatementType::LET<Vi>&) {
			visitExported(vi, var.exported);
			visitPat(vi, var.names);
			visitExpList(vi, var.exprs);
		},
		varcase(parse::StatementType::CONST<Vi>&) {
			visitExported(vi, var.exported);
			visitPat(vi, var.names);
			visitExpList(vi, var.exprs);
		},
		varcase(parse::StatementType::FUNC_CALL<Vi>&) {
			visitLimPrefixExpr(vi, *var.val);
			visitArgChain(vi, var.argChain);
		},
		varcase(parse::StatementType::BLOCK<Vi>&) {
			visitBlock(vi, var.bl);
		},
		varcase(parse::StatementType::GOTO<Vi>&) {
			//TODO
		},
		varcase(parse::StatementType::BREAK&) {
			//TODO
		},
		varcase(parse::StatementType::LABEL<Vi>&) {
			//TODO
		},
		varcase(parse::StatementType::USE&) {
			//TODO
		},
		varcase(parse::StatementType::SEMICOLON&) {
			//TODO
		},
		varcase(parse::StatementType::IfCond<Vi>&) {
			visitExpr(vi, *var.cond);
			visitSoe(vi, *var.bl);
			for (auto& [cond, soe] : var.elseIfs)
			{
				visitExpr(vi, cond);
				visitSoe(vi, soe);
			}
			if (var.elseBlock.has_value())
				visitSoe(vi, **var.elseBlock);
		},
		varcase(parse::StatementType::WHILE_LOOP<Vi>&) {
			visitExpr(vi, var.cond);
			visitBlock(vi, var.bl);
		},
		varcase(parse::StatementType::REPEAT_UNTIL<Vi>&) {
			visitBlock(vi, var.bl);
			visitExpr(vi, var.cond);
		},
		varcase(parse::StatementType::FOR_LOOP_NUMERIC<Vi>&) {
			visitPat(vi, var.varName);
			visitExpr(vi, var.start);
			visitExpr(vi, var.end);
			if (var.step.has_value())
				visitExpr(vi, *var.step);
			visitBlock(vi, var.bl);
		},
		varcase(parse::StatementType::FOR_LOOP_GENERIC<Vi>&) {
			visitPat(vi, var.varNames);
			visitExpr(vi, var.exprs);
			visitBlock(vi, var.bl);
		},
		varcase(parse::StatementType::Struct<Vi>&) {
			visitExported(vi, var.exported);
			visitName(vi, var.name);
			visitParams(vi, var.params);
			visitTypeExp(vi, var.type);
		},
		varcase(parse::StatementType::Union<Vi>&) {
			visitExported(vi, var.exported);
			visitName(vi, var.name);
			visitParams(vi, var.params);
			visitTable(vi, var.type);
		},
		varcase(parse::StatementType::FUNCTION_DEFv<Vi::settings()&parse::sluSyn>&) {
			visitExported(vi, var.exported);
			visitSafety(vi, var.func.safety);
			visitName(vi, var.name);
			visitBlock(vi, var.func.block);
			if (var.func.retType.has_value())
				visitTypeExp(vi, *var.func.retType);
			visitParams(vi, var.func.params);
		},
		varcase(parse::StatementType::FunctionDecl<Vi>&) {
			visitExported(vi, var.exported);
			visitSafety(vi, var.safety);
			visitName(vi, var.name);
			if (var.retType.has_value())
				visitTypeExp(vi, *var.retType);
			visitParams(vi, var.params);
		},
		varcase(parse::StatementType::ExternBlock<Vi>&) {
			visitSafety(vi, var.safety);
			visitString(vi, var.abi);
			visitBlock(vi, var.bl);
		},
		varcase(parse::StatementType::UnsafeBlock<Vi>&) {
			visitBlock(vi, var.bl);
		},
		varcase(const parse::StatementType::UNSAFE_LABEL) {
			//TODO
		},
		varcase(const parse::StatementType::SAFE_LABEL) {
			//TODO
		},
		varcase(parse::StatementType::DROP<Vi>&) {
			visitExpr(vi, var.expr);
		},
		varcase(parse::StatementType::MOD_DEF<Vi>&) {
			visitExported(vi, var.exported);
			visitName(vi, var.name);
			//TODO
		},
		varcase(parse::StatementType::MOD_DEF_INLINE<Vi>&) {
			visitExported(vi, var.exported);
			visitName(vi, var.name);
			visitBlock(vi, var.bl);
		}
		);
		Slu_CALL_VISIT_FN_POST(Stat);
	}
	template<AnyVisitor Vi>
	inline void visitSoe(Vi& vi, parse::Soe<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Soe);
		ezmatch(itm)(
		varcase(parse::SoeType::BLOCK<Vi>&) {
			visitBlock(vi, var);
		},
		varcase(parse::SoeType::EXPR<Vi>&) {
			visitExpr(vi, *var);
		}
		);
		Slu_CALL_VISIT_FN_POST(Soe);
	}
	template<AnyVisitor Vi>
	inline void visitBlock(Vi& vi, parse::Block<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Block);
		for (auto& i : itm.statList)
			visitStat(vi, i);
		if (itm.hadReturn)
		{
			Slu_CALL_VISIT_FN_PRE(BlockReturn);
			visitExpList(vi, itm.retExprs);
		}
		Slu_CALL_VISIT_FN_POST(Block);
	}
	template<AnyVisitor Vi>
	void visitFile(Vi& vi,parse::ParsedFile<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(File);
		visitBlock(vi, itm.code);
		Slu_CALL_VISIT_FN_POST(File);
	}
}
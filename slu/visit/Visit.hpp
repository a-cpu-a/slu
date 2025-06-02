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
#include <slu/visit/AnyVisitor.hpp>

namespace slu::visit
{
#define Slu_CALL_VISIT_FN_PRE(_Name) \
	if constexpr(requires{vi.pre##_Name(itm);}) \
		if(vi.pre ## _Name (itm)) \
			return

#define Slu_CALL_VISIT_FN_SEP(_Name,_i,_vec) \
	if constexpr(requires{vi.sep##_Name(_i);}) \
		if(&_i != &_vec.back()) \
			vi.sep##_Name(_i)

#define Slu_CALL_VISIT_FN_POST(_Name) \
	if constexpr(requires{vi.post##_Name(itm);}) \
		vi.post##_Name(itm)

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
		varcase(parse::DestrSpecType::Prefix&) {}
		);
		Slu_CALL_VISIT_FN_POST(DestrSpec);
	}
	template<AnyVisitor Vi>
	inline void visitPat(Vi& vi, parse::Pat<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Pat);
		ezmatch(itm)(
		varcase(const parse::PatType::DestrAny) {},

		varcase(parse::PatType::Simple<Vi>&) {
			visitExpr(vi, var);
		},

		varcase(parse::PatType::DestrFields<Vi>&) {
			visitDestrSpec(vi, var.spec);
			for (auto& i : var.items)
				visitPat(vi, i.pat);
		},
		varcase(parse::PatType::DestrList<Vi>&) {
			visitDestrSpec(vi, var.spec);
			for (auto& i : var.items)
				visitPat(vi, i);
		},

		varcase(parse::PatType::DestrName<Vi>&) {
			visitDestrSpec(vi, var.spec);
		},
		varcase(parse::PatType::DestrNameRestrict<Vi>&) {
			visitDestrSpec(vi, var.spec);
			visitExpr(vi, var.restriction);
		}
		);
		Slu_CALL_VISIT_FN_POST(Pat);
	}
	template<AnyVisitor Vi>
	inline void visitVar(Vi& vi, parse::Var<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Var);
		ezmatch(itm.base)(
		varcase(parse::BaseVarType::EXPR<Vi>&) { visitExpr(vi, var.start); },
		varcase(parse::BaseVarType::NAME<Vi>&) {},
		varcase(const parse::BaseVarType::Root) {}
		);
		for (auto& i : itm.sub)
		{
			visitArgChain(i.funcCalls);
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
			Slu_CALL_VISIT_FN_SEP(VarList,i,itm);
			visitVar(i);
		}
		Slu_CALL_VISIT_FN_POST(VarList);
	}
	template<AnyVisitor Vi>
	inline void visitTypeExp(Vi& vi, parse::TypeExpr& itm)
	{
		Slu_CALL_VISIT_FN_PRE(TypeExp);
		//TODO
		Slu_CALL_VISIT_FN_POST(TypeExp);
	}
	template<AnyVisitor Vi>
	inline void visitExpr(Vi& vi, parse::Expression<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Expr);
		//TODO
		Slu_CALL_VISIT_FN_POST(Expr);
	}
	template<AnyVisitor Vi>
	inline void visitExpList(Vi& vi, parse::ExpList<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(ExpList);
		for (auto& i : itm)
		{
			Slu_CALL_VISIT_FN_SEP(ExpList,i,itm);
			visitExpr(vi, i);
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
	template<AnyVisitor Vi>
	inline void visitParams(Vi& vi, parse::ParamList<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Params);
		for (auto& i : itm)
		{
			Slu_CALL_VISIT_FN_SEP(Params, i, itm);
			visitPat(vi, i.name);
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
			visitExpList(vi, var.exprs);
			visitPat(vi, var.names);
		},
		varcase(parse::StatementType::LET<Vi>&) {
			visitExpList(vi, var.exprs);
			visitPat(vi, var.names);
		},
		varcase(parse::StatementType::CONST<Vi>&) {
			visitExpList(vi, var.exprs);
			visitPat(vi, var.names);
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
		varcase(parse::StatementType::MOD_DEF<Vi>&) {
			//TODO
		},
		varcase(parse::StatementType::SEMICOLON&) {
			//TODO
		},
		varcase(parse::StatementType::IfCond<Vi>&) {
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
		varcase(parse::StatementType::WHILE_LOOP<Vi>&) {
			visitBlock(vi, var.bl);
			visitExpr(vi, var.cond);
		},
		varcase(parse::StatementType::REPEAT_UNTIL<Vi>&) {
			visitBlock(vi, var.bl);
			visitExpr(vi, var.cond);
		},
		varcase(parse::StatementType::FOR_LOOP_NUMERIC<Vi>&) {
			visitBlock(vi, var.bl);
			visitExpr(vi, var.start);
			visitExpr(vi, var.end);
			if (var.step.has_value())
				visitExpr(vi, *var.step);
			visitPat(vi, var.varName);
		},
		varcase(parse::StatementType::FOR_LOOP_GENERIC<Vi>&) {
			visitBlock(vi, var.bl);
			visitExpr(vi, var.exprs);
			visitPat(vi, var.varNames);
		},
		varcase(parse::StatementType::Struct<Vi>&) {
			visitParams(vi, var.params);
			visitTypeExp(vi, var.type);
		},
		varcase(parse::StatementType::Union<Vi>&) {
			visitParams(vi, var.params);
			visitTable(vi, var.type);
		},
		varcase(parse::StatementType::FuncDefBase<Vi>&) {
			visitBlock(vi, var.func.block);
			if (var.func.retType.has_value())
				visitTypeExp(vi, *var.func.retType);
			visitParams(vi, var.func.params);
		},
		varcase(parse::StatementType::FunctionDecl<Vi>&) {
			if (var.retType.has_value())
				visitTypeExp(vi, *var.retType);
			visitParams(vi, var.params);
		},
		varcase(parse::StatementType::ExternBlock<Vi>&) {
			//TODO
		},
		varcase(parse::StatementType::UnsafeBlock<Vi>&) {
			//TODO
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
		varcase(parse::StatementType::MOD_DEF_INLINE<Vi>&) {
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
			visitExpList(vi, itm.retExprs);
		Slu_CALL_VISIT_FN_POST(Block);
	}
	template<AnyVisitor Vi>
	void visitFile(Vi& vi,parse::ParsedFile<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(File);
		visitBlock(vi, vi,itm.code);
		Slu_CALL_VISIT_FN_POST(File);
	}
}
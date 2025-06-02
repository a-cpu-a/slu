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
#define Slu_CALL_VISIT_FN_PRE(_Name) \
	if constexpr(requires{vi.pre##_Name(itm);}) \
		if(vi.pre ## _Name (itm)) \
			return

#define Slu_CALL_VISIT_FN_SEP(_Name,_i,_vec) \
	if constexpr(requires{vi.sep##_Name(_vec,_i);}) \
		if(&_i != &_vec.back()) \
			vi.sep##_Name(_vec,_i)

#define Slu_CALL_VISIT_FN_POST(_Name) \
	if constexpr(requires{vi.post##_Name(itm);}) \
		vi.post##_Name(itm)

	template<AnyVisitor Vi>
	inline void visitString(Vi& vi, std::string_view itm)
	{
		Slu_CALL_VISIT_FN_PRE(String);
	}
	template<AnyVisitor Vi>
	inline void visitName(Vi& vi, parse::MpItmId<Vi>& itm)
	{
		Slu_CALL_VISIT_FN_PRE(Name);
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
			//TODO
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
			Slu_CALL_VISIT_FN_PRE(DestrAny);
		},

		varcase(parse::PatType::Simple<Vi>&) {
			Slu_CALL_VISIT_FN_PRE(DestrSimple);
			visitExpr(vi, var);
			Slu_CALL_VISIT_FN_POST(DestrSimple);
		},

		varcase(parse::PatType::DestrFields<Vi>&) {
			Slu_CALL_VISIT_FN_PRE(DestrFields);
			visitDestrSpec(vi, var.spec);
			Slu_CALL_VISIT_FN_PRE(DestrFieldsFirst);
			for (auto& i : var.items)
			{
				Slu_CALL_VISIT_FN_PRE(DestrField);
				visitName(vi, i.name);
				Slu_CALL_VISIT_FN_PRE(DestrFieldPat);
				visitPat(vi, i.pat);
				Slu_CALL_VISIT_FN_POST(DestrField);
				Slu_CALL_VISIT_FN_SEP(DestrFields, i, itm);
			}
			if(!var.name.empty())
			{
				Slu_CALL_VISIT_FN_PRE(DestrFieldsName);
				visitName(vi, var.name);
			}
			Slu_CALL_VISIT_FN_POST(DestrFields);
		},
		varcase(parse::PatType::DestrList<Vi>&) {
			Slu_CALL_VISIT_FN_PRE(DestrList);
			visitDestrSpec(vi, var.spec);
			Slu_CALL_VISIT_FN_PRE(DestrListFirst);
			for (auto& i : var.items)
			{
				visitPat(vi, i);
				Slu_CALL_VISIT_FN_SEP(DestrList, i, itm);
			}
			if (!var.name.empty())
			{
				Slu_CALL_VISIT_FN_PRE(DestrListName);
				visitName(vi, var.name);
			}
			Slu_CALL_VISIT_FN_POST(DestrList);
		},

		varcase(parse::PatType::DestrName<Vi>&) {
			Slu_CALL_VISIT_FN_PRE(DestrName);
			visitDestrSpec(vi, var.spec);
			if (!var.name.empty())
			{
				Slu_CALL_VISIT_FN_PRE(DestrNameName);
				visitName(vi, var.name);
			}
			Slu_CALL_VISIT_FN_POST(DestrName);
		},
		varcase(parse::PatType::DestrNameRestrict<Vi>&) {
			Slu_CALL_VISIT_FN_PRE(DestrNameRestrict);
			visitDestrSpec(vi, var.spec);
			if (!var.name.empty())
			{
				Slu_CALL_VISIT_FN_PRE(DestrNameRestrictName);
				visitName(vi, var.name);
			}
			Slu_CALL_VISIT_FN_PRE(DestrNameRestriction);
			visitExpr(vi, var.restriction);
			Slu_CALL_VISIT_FN_POST(DestrNameRestrict);
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
			Slu_CALL_VISIT_FN_PRE(BaseVarExpr);
			visitExpr(vi, var.start);
			Slu_CALL_VISIT_FN_POST(BaseVarExpr);
		},
		varcase(parse::BaseVarType::NAME<Vi>&) {
			Slu_CALL_VISIT_FN_PRE(BaseVarName);
			visitName(vi, var.v);
			Slu_CALL_VISIT_FN_POST(BaseVarName);
		},
		varcase(const parse::BaseVarType::Root) {
			Slu_CALL_VISIT_FN_PRE(BaseVarRoot);
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
		varcase(parse::StatementType::FuncDefBase<Vi::settings()&parse::sluSyn>&) {
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
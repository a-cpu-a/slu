/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Parse.hpp>
#include <slua/ext/CppMatch.hpp>
#include <slua/generator/Output.hpp>


namespace sluaParse
{
	inline void genExpr(AnyOutput auto& out, const Expression& obj)
	{
		ezmatch(obj.data)(
		varcase(ExprType::NIL) {
			out.add("nil");
		},
		varcase(ExprType::FALSE) {
			out.add("false");
		},
		varcase(ExprType::TRUE) {
			out.add("true");
		},
		varcase(ExprType::VARARGS) {
			out.add("...");
		},
		varcase(ExprType::FUNCTION_DEF) {
			genFuncDef(out, var);
		},
		varcase(ExprType::FUNC_CALL) {
			genVar(out, var.v);
		},
		varcase(ExprType::LIM_PREFIX_EXP) {
			genVar(out, var.v);
		},
		varcase(ExprType::TABLE_CONSTRUCTOR) {
			genTableConstructor(out, var.v);
		},
		varcase(ExprType::MULTI_OPERATION) {
			genExpr(out, var.left);
			out.add(' ')
				.add(var.op)
				.add(' ');
			genExpr(out, var.right);
		},
		);
	}

	inline void genExpList(AnyOutput auto& out, const ExpList& obj)
	{
		for (const Expression& e : obj)
		{
			genExpr(out, e);
			if (&e != &obj.back())
				out.add(", ");
		}
	}
	inline void genLiteral(AnyOutput auto& out, const std::string& obj)
	{
		//TODO: escape, quote.
		out.add(obj);
	}

	inline void genArgFuncCall(AnyOutput auto& out, const ArgFuncCall& arg)
	{
		if (!arg.funcName.empty())
		{
			out.add(':')
				.add(arg.funcName);
		}
		ezmatch(arg.args)(
			varcase(ArgsType::EXPLIST) {
			out.add('(');
			genExpList(out, var.v);
			out.add(')');
		},
			varcase(ArgsType::TABLE) {
			genTableConstructor(out, var.v);
		},
			varcase(ArgsType::LITERAL) {
			genLiteral(out, var.v);
		}
		);
	}

	inline void genSubVar(AnyOutput auto& out, const SubVar& obj)
	{
		ezmatch(obj)(
			varcase(SubVarType::EXPR) {
			out.add('[');
			genExpr(out, var);
			out.add(']');
		},
			varcase(SubVarType::NAME) {
			out.add('.')
				.add(var);
		}
		);
		for (const ArgFuncCall& arg : obj.funcCalls)
		{
			genArgFuncCall(out, arg);
		}
	}

	inline void genVar(AnyOutput auto& out, const Var& obj)
	{
		ezmatch(obj.base)(
			varcase(BaseVarType::NAME) {
			out.add(var);
		},
			varcase(BaseVarType::EXPR) {
			genExpr(out, var.start);
			genSubVar(out, var.sub);
		}
		);
		for (const SubVar& sub :  obj.sub)
		{
			genSubVar(out, sub);
		}
	}
	inline void genFuncDef(AnyOutput auto& out, const StatementType::FUNCTION_DEF& var)
	{
		out.add("function ")
			.add(var.name)
			.add('(');

		for (const Parameter& par : var.func.params)
		{
			out.add(par.name);
			if (&par != &var.func.params.back())
				out.add(", ");

		}
		out.add(')')
			.tabUpNewl();

		genBlock(out, var.func.block);

		out.unTabNewl()
			.addNewl("end");

		out.newLine();//Extra spacing
	}

	inline void genVarList(AnyOutput auto& out, const std::vector<Var>& obj)
	{
		for (const Var& v : obj)
		{
			genVar(out, v);
			if (&v != &obj.back())
				out.add(", ");
		}
	}
	inline void genAtribNameList(AnyOutput auto& out, const AttribNameList& obj)
	{
		for (const AttribName& v : obj)
		{
			out.add(v.name);
			if (!v.attrib.empty())
				out.add(" <")
				.add(v.attrib)
				.add('>');
			if (&v != &obj.back())
				out.add(", ");
		}
	}
	inline void genNames(AnyOutput auto& out, const NameList& obj)
	{
		for (const std::string& v : obj)
		{
			out.add(v);
			if (&v != &obj.back())
				out.add(", ");
		}
	}

	inline void genStat(AnyOutput auto& out, const Statement& obj)
	{
		ezmatch(obj.data)(

			varcase(StatementType::NONE) { _ASSERT(false); },

		varcase(StatementType::SEMICOLON) {
			out.add(';');//TODO: should elide duplicate semicolons?
		},

		varcase(StatementType::ASSIGN) {
			genVarList(out, var.vars);
			out.add(" = ");
			genExpList(out, var.exprs);
			out.addNewl(';');
		},
		varcase(StatementType::LOCAL_ASSIGN) {
			out.add("local ");
			genAtribNameList(out,var.names);
			out.add(" = ");
			genExpList(out, var.exprs);
			out.addNewl(';');
		},

		varcase(StatementType::FUNC_CALL) {
			ezmatch(var.val)(
			ezcase(varObj,LimPrefixExprType::VAR) {
				genVar(out,varObj.v);
			},
			ezcase(expr,LimPrefixExprType::EXPR) {
				out.add('(');
				genExpr(out, expr.v);
				out.add(')');
			}
			);
			for (const ArgFuncCall& arg : var.argChain)
			{
				genArgFuncCall(out, arg);
			}
			out.addNewl(';');
		},
		varcase(StatementType::LABEL) {
			out.unTabTemp()
				.add("::")
				.add(var.v)
				.addNewl("::")
				.tabUpTemp();
		},
		varcase(StatementType::BREAK) {
			out.addNewl("break;");
		},
		varcase(StatementType::GOTO) {
			out.add("goto ")
				.add(var.v)
				.addNewl(';');
		},
		varcase(StatementType::DO_BLOCK) {
			out.newLine();//Extra spacing
			out.add("do")
				.tabUpNewl();
			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl("end");
		},
		varcase(StatementType::WHILE_LOOP) {
			out.newLine();//Extra spacing
			out.add("while ");
			genExpr(out, var.cond);
			out.add(" do")
				.tabUpNewl();
			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl("end");
		},
		varcase(StatementType::REPEAT_UNTIL) {
			out.add("repeat")
				.tabUpNewl();
			genBlock(out, var.bl);
			out.unTabNewl()
				.add("until ");
			genExpr(out, var.cond);
			out.addNewl(';');
			out.newLine();//Extra spacing
		},

		varcase(StatementType::IF_THEN_ELSE) {
			out.add("if ");
			genExpr(out, var.cond);
			out.add(" then")
				.tabUpNewl();
			genBlock(out, var.bl);

			if (!var.elseIfs.empty())
			{
				for (const auto& [expr, bl] : var.elseIfs)
				{
					out.unTabNewl()
						.add("elseif ");
					genExpr(out, expr);
					out.add(" then")
						.tabUpNewl();
					genBlock(out, bl);
				}
			}
			if (var.elseBlock)
			{
				out.unTabNewl()
					.add("else")
					.tabUpNewl();
				genBlock(out, *var.elseBlock);
			}
			out.unTabNewl()
				.addNewl("end");
		},

		varcase(StatementType::FOR_LOOP_NUMERIC) {
			out.add("for ")
				.add(var.varName)
				.add(" = ");
			genExpr(out, var.start);
			out.add(", ");
			genExpr(out, var.end);
			if (var.step)
			{
				out.add(", ");
				genExpr(out, *var.step);
			}
			out.add(" do")
				.tabUpNewl();
			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl("end");
		},
		varcase(StatementType::FOR_LOOP_GENERIC) {
			out.add("for ");
			genNames(out, var.varNames);
			out.add(" in ");
			genExpList(out, var.exprs);
			out.add(" do")
				.tabUpNewl();
			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl("end");
		},

		varcase(StatementType::FUNCTION_DEF) {
			genFuncDef(out, var);
		},
		varcase(StatementType::LOCAL_FUNCTION_DEF) {
			out.add("local ");
			genFuncDef(out, var);
		}

		);
	}

	inline void genBlock(AnyOutput auto& out, const Block& obj)
	{
		for (const Statement& s : obj.statList)
		{
			genStat(out, s);
		}

		if (obj.hadReturn)
		{
			out.add("return");
			if (!obj.retExprs.empty())
			{
				out.add(" ");
				genExpList(out, obj.retExprs);
			}
			out.add(";");
		}
	}

	inline void genFile(AnyOutput auto& out,const ParsedFile& obj)
	{
		genBlock(obj.code, out);
	}
}
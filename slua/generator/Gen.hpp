/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>
#include <string_view>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Parse.hpp>
#include <slua/ext/CppMatch.hpp>
#include <slua/generator/Output.hpp>


namespace sluaParse
{
	inline std::string_view getBinOpAsStr(const BinOpType t)
	{
		using namespace std::literals;
		switch (t)
		{
		case BinOpType::ADD:
			return "+"sv;
		case BinOpType::SUBTRACT:
			return "-"sv;
		case BinOpType::MULTIPLY:
			return "*"sv;
		case BinOpType::DIVIDE:
			return "/"sv;
		case BinOpType::FLOOR_DIVIDE:
			return "//"sv;
		case BinOpType::EXPONENT:
			return "^"sv;
		case BinOpType::MODULO:
			return "%"sv;
		case BinOpType::BITWISE_AND:
			return "&"sv;
		case BinOpType::BITWISE_XOR:
			return "~"sv;
		case BinOpType::BITWISE_OR:
			return "|"sv;
		case BinOpType::SHIFT_RIGHT:
			return ">>"sv;
		case BinOpType::SHIFT_LEFT:
			return "<<"sv;
		case BinOpType::CONCATENATE:
			return ".."sv;
		case BinOpType::LESS_THAN:
			return "<"sv;
		case BinOpType::LESS_EQUAL:
			return "<="sv;
		case BinOpType::GREATER_THAN:
			return ">"sv;
		case BinOpType::GREATER_EQUAL:
			return ">="sv;
		case BinOpType::EQUAL:
			return "=="sv;
		case BinOpType::NOT_EQUAL:
			return "~="sv;
		case BinOpType::LOGICAL_AND:
			return "and"sv;
		case BinOpType::LOGICAL_OR:
			return "or"sv;
		default:
			_ASSERT(false);
			return "<ERROR>"sv;
		}
	}
	inline std::string_view getUnOpAsStr(const UnOpType t)
	{
		using namespace std::literals;
		switch (t)
		{
		case UnOpType::NEGATE:
			return " -"sv;//TODO: elide space, when there is one already
		case UnOpType::LOGICAL_NOT:
			return " not "sv;
		case UnOpType::LENGTH:
			return " #"sv;
		case UnOpType::BITWISE_NOT:
			return " ~"sv;
		default:
			_ASSERT(false);
			return "<ERROR>"sv;
		}
	}

	inline void genTableConstructor(AnyOutput auto& out, const TableConstructor& obj)
	{
		out.add('{')
			.tabUpNewl();

		for (const Field& f : obj)
		{
			ezmatch(f)(
			varcase(FieldType::EXPR2EXPR) {
				out.add('[');
				genExpr(out, var.idx);
				out.add("] = ");
				genExpr(out, var.v);
			},
			varcase(FieldType::NAME2EXPR) {
				out.add(var.idx)
					.add(" = ");
				genExpr(out, var.v);
			},
			varcase(FieldType::EXPR) {
				genExpr(out, var.v);
			}
			);
			out.addNewl(',');
		}

		out.add('}')
			.unTabNewl();
	}

	inline void genLimPrefixExpr(AnyOutput auto& out, const LimPrefixExpr& obj)
	{
		ezmatch(obj)(
		varcase(LimPrefixExprType::VAR) {
			genVar(out, var.v);
		},
		varcase(LimPrefixExprType::EXPR) {
			out.add('(');
			genExpr(out, var.v);
			out.add(')');
		}
		);
	}

	inline void genFuncCall(AnyOutput auto& out,const FuncCall& obj)
	{
		genLimPrefixExpr(out, *obj.val);
		for (const ArgFuncCall& arg : obj.argChain)
		{
			genArgFuncCall(out, arg);
		}
	}

	inline void genExpr(AnyOutput auto& out, const Expression& obj)
	{
		for (const UnOpType t : obj.unOps)
		{
			out.add(getUnOpAsStr(t));
		}
		using namespace std::literals;
		ezmatch(obj.data)(
		varcase(ExprType::NIL) {
			out.add("nil"sv);
		},
		varcase(ExprType::FALSE) {
			out.add("false"sv);
		},
		varcase(ExprType::TRUE) {
			out.add("true"sv);
		},
		varcase(ExprType::VARARGS) {
			out.add("..."sv);
		},
		varcase(ExprType::FUNCTION_DEF) {
			genFuncDef(out, var.v,""sv);
		},
		varcase(ExprType::FUNC_CALL) {
			genFuncCall(out, var);
		},
		varcase(ExprType::LIM_PREFIX_EXP) {
			genLimPrefixExpr(out, *var);
		},
		varcase(ExprType::TABLE_CONSTRUCTOR) {
			genTableConstructor(out, var.v);
		},
		varcase(ExprType::MULTI_OPERATION) {
			genExpr(out, *var.first);
			for (const auto& [op,ex] : var.extra)
			{
				out.add(' ')
					.add(getBinOpAsStr(op))
					.add(' ');
				genExpr(out, ex);
			}
		}
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
		out.add('"');
		for (const char ch : obj)
		{
			switch (ch)
			{
			case '\n': out.add("\\n"); break;
			case '\r': out.add("\\r"); break;
			case '\t': out.add("\\t"); break;
			case '\b': out.add("\\b");	break;
			case '\a': out.add("\\a"); break;
			case '\f': out.add("\\f"); break;
			case '\v': out.add("\\v"); break;
			case '"': out.add("\\\""); break;
			case '\\': out.add("\\\\"); break;
			case '\0': out.add("\\x00"); break;
			default:
				out.add(ch);
				break;
			}
		}
		out.add('"');
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
		ezmatch(obj.idx)(
		varcase(SubVarType::EXPR) {
			out.add('[');
			genExpr(out, var.idx);
			out.add(']');
		},
		varcase(SubVarType::NAME) {
			out.add('.')
				.add(var.idx);
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
	inline void genFuncDef(AnyOutput auto& out, const Function& var,const std::string_view name)
	{
		out.add("function ")
			.add(name)
			.add('(');

		for (const Parameter& par : var.params)
		{
			out.add(par.name);
			if (&par != &var.params.back() || var.hasVarArgParam)
				out.add(", ");

		}
		if (var.hasVarArgParam)
			out.add("...");

		out.add(')')
			.tabUpNewl();

		genBlock(out, var.block);

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
			genFuncCall(out, var);
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
			genFuncDef(out, var.func,var.name);
		},
		varcase(StatementType::LOCAL_FUNCTION_DEF) {
			out.add("local ");
			genFuncDef(out, var.func, var.name);
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
		genBlock(out, obj.code);
	}
}
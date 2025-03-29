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
			.template tabUpNewl<false>();

		for (const Field& f : obj)
		{
			ezmatch(f)(
			varcase(const FieldType::NONE) { _ASSERT(false); },

			varcase(const FieldType::EXPR2EXPR&) {
				out.addIndent();
				out.add('[');
				genExpr(out, var.idx);
				out.add("] = ");
				genExpr(out, var.v);
			},
			varcase(const FieldType::NAME2EXPR&) {
				out.addIndent();
				out.add(var.idx)
					.add(" = ");
				genExpr(out, var.v);
			},
			varcase(const FieldType::EXPR&) {
				out.addIndent();
				genExpr(out, var.v);
			}
			);
			out.template addNewl<false>(',');
		}

		out.unTab()
			.add('}');
	}

	inline void genLimPrefixExpr(AnyOutput auto& out, const LimPrefixExpr& obj)
	{
		ezmatch(obj)(
		varcase(const LimPrefixExprType::VAR&) {
			genVar(out, var.v);
		},
		varcase(const LimPrefixExprType::EXPR&) {
			out.add('(');
			genExpr(out, var.v);
			out.add(')');
		}
		);
	}

	template<AnyOutput Out>
	inline void genFuncCall(Out& out,const FuncCall<Out>& obj)
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
		varcase(const ExprType::NIL) {
			out.add("nil"sv);
		},
		varcase(const ExprType::FALSE) {
			out.add("false"sv);
		},
		varcase(const ExprType::TRUE) {
			out.add("true"sv);
		},
		varcase(const ExprType::VARARGS) {
			out.add("..."sv);
		},
		varcase(const ExprType::NUMERAL) {
			if (isinf(var.v) && var.v>0.0f)
				out.add("1e999");
			else
				out.add(std::to_string(var.v));
		},
		varcase(const ExprType::NUMERAL_I64) {
			if (var.v < 0)
			{
				out.add("0x");
				for (size_t i = 0; i < 16; i++)
				{
					const uint8_t c = (uint64_t(var.v) >> (60 - 4 * i)) & 0xF;
					if(c<=9)
						out.add('0' + c);
					else
						out.add('A' + (c - 10));
				}
			}
			else
				out.add(std::to_string(var.v));
		},
		varcase(const ExprType::LITERAL_STRING&) {
			genLiteral(out,var.v);
		},
		varcase(const ExprType::FUNCTION_DEF&) {
			genFuncDef(out, var.v,""sv);
		},
		varcase(const ExprType::FUNC_CALL&) {
			genFuncCall(out, var);
		},
		varcase(const ExprType::LIM_PREFIX_EXP&) {
			genLimPrefixExpr(out, *var);
		},
		varcase(const ExprType::TABLE_CONSTRUCTOR&) {
			genTableConstructor(out, var.v);
		},
		varcase(const ExprType::MULTI_OPERATION&) {
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
		varcase(const ArgsType::EXPLIST&) {
			out.add('(');
			genExpList(out, var.v);
			out.add(')');
		},
		varcase(const ArgsType::TABLE&) {
			genTableConstructor(out, var.v);
		},
		varcase(const ArgsType::LITERAL&) {
			genLiteral(out, var.v);
		}
		);
	}

	inline void genSubVar(AnyOutput auto& out, const SubVar& obj)
	{
		for (const ArgFuncCall& arg : obj.funcCalls)
		{
			genArgFuncCall(out, arg);
		}
		ezmatch(obj.idx)(
		varcase(const SubVarType::EXPR&) {
			out.add('[');
			genExpr(out, var.idx);
			out.add(']');
		},
		varcase(const SubVarType::NAME&) {
			if (var.idx.empty())return;
			out.add('.')
				.add(var.idx);
		}
		);
	}

	inline void genVar(AnyOutput auto& out, const Var& obj)
	{
		ezmatch(obj.base)(
		varcase(const BaseVarType::NAME&) {
			out.add(var);
		},
		varcase(const BaseVarType::EXPR&) {
			out.add('(');
			genExpr(out, var.start);
			out.add(')');
			genSubVar(out, var.sub);
		}
		);
		for (const SubVar& sub :  obj.sub)
		{
			genSubVar(out, sub);
		}
	}
	template<AnyOutput Out>
	inline void genFuncDef(Out& out, const Function<Out>& var,const std::string_view name)
	{
		out.add("function ")
			.add(name)
			.add('(');

		for (const Parameter<Out>& par : var.params)
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

		varcase(const StatementType::SEMICOLON) {
			if(!out.wasSemicolon)
				out.add(';');
			out.wasSemicolon = true;
		},

		varcase(const StatementType::ASSIGN&) {
			genVarList(out, var.vars);
			out.add(" = ");
			genExpList(out, var.exprs);
			out.addNewl(';');
			out.wasSemicolon = true;
		},
		varcase(const StatementType::LOCAL_ASSIGN&) {
			out.add("local ");
			genAtribNameList(out,var.names);
			if (!var.exprs.empty())
			{
				out.add(" = ");
				genExpList(out, var.exprs);
			}
			out.addNewl(';');
			out.wasSemicolon = true;
		},

		varcase(const StatementType::FUNC_CALL&) {
			genFuncCall(out, var);
			out.addNewl(';');
			out.wasSemicolon = true;
		},
		varcase(const StatementType::LABEL&) {
			out.unTabTemp()
				.add("::")
				.add(var.v)
				.addNewl("::")
				.tabUpTemp();
		},
		varcase(const StatementType::BREAK) {
			out.addNewl("break;");
			out.wasSemicolon = true;
		},
		varcase(const StatementType::GOTO&) {
			out.add("goto ")
				.add(var.v)
				.addNewl(';');
			out.wasSemicolon = true;
		},
		varcase(const StatementType::DO_BLOCK&) {
			out.newLine();//Extra spacing
			out.add("do")
				.tabUpNewl();
			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl("end");
		},
		varcase(const StatementType::WHILE_LOOP&) {
			out.newLine();//Extra spacing
			out.add("while ");
			genExpr(out, var.cond);
			out.add(" do")
				.tabUpNewl();
			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl("end");
		},
		varcase(const StatementType::REPEAT_UNTIL&) {
			out.add("repeat")
				.tabUpNewl();
			genBlock(out, var.bl);
			out.unTabNewl()
				.add("until ");
			genExpr(out, var.cond);
			out.addNewl(';');
			out.newLine();//Extra spacing
			out.wasSemicolon = true;
		},

		varcase(const StatementType::IF_THEN_ELSE&) {
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

		varcase(const StatementType::FOR_LOOP_NUMERIC&) {
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
		varcase(const StatementType::FOR_LOOP_GENERIC&) {
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

		varcase(const StatementType::FUNCTION_DEF&) {
			genFuncDef(out, var.func,var.name);
		},
		varcase(const StatementType::LOCAL_FUNCTION_DEF&) {
			out.add("local ");
			genFuncDef(out, var.func, var.name);
		}

		);
	}

	template<AnyOutput Out>
	inline void genBlock(Out& out, const Block<Out>& obj)
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
			out.wasSemicolon = true;
		}
	}

	inline void genFile(AnyOutput auto& out,const ParsedFile& obj)
	{
		genBlock(out, obj.code);
	}
}
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

	template<AnyOutput Out>
	inline void genTableConstructor(Out& out, const TableConstructor<Out>& obj)
	{
		out.add('{')
			.template tabUpNewl<false>();

		for (const Field<Out>& f : obj)
		{
			ezmatch(f)(
			varcase(const FieldType::NONE) { _ASSERT(false); },

			varcase(const FieldType::EXPR2EXPR<Out>&) {
				out.addIndent();
				out.add('[');
				genExpr(out, var.idx);
				out.add("] = ");
				genExpr(out, var.v);
			},
			varcase(const FieldType::NAME2EXPR<Out>&) {
				out.addIndent();
				out.add(var.idx)
					.add(" = ");
				genExpr(out, var.v);
			},
			varcase(const FieldType::EXPR<Out>&) {
				out.addIndent();
				genExpr(out, var.v);
			}
			);
			out.template addNewl<false>(',');
		}

		out.unTab()
			.add('}');
	}
	template<AnyOutput Out>
	inline void genLimPrefixExpr(Out& out, const LimPrefixExpr<Out>& obj)
	{
		ezmatch(obj)(
		varcase(const LimPrefixExprType::VAR<Out>&) {
			genVar(out, var.v);
		},
		varcase(const LimPrefixExprType::EXPR<Out>&) {
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
		for (const ArgFuncCall<Out>& arg : obj.argChain)
		{
			genArgFuncCall(out, arg);
		}
	}
	inline void writeU64Hex(AnyOutput auto& out, const uint64_t v) {
		for (size_t i = 0; i < 16; i++)
		{
			const uint8_t c = (uint64_t(v) >> (60 - 4 * i)) & 0xF;
			if (c <= 9)
				out.add('0' + c);
			else
				out.add('A' + (c - 10));
		}
	}

	template<AnyOutput Out>
	inline void genExpr(Out& out, const Expression<Out>& obj)
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
				writeU64Hex(out,var.v);
			}
			else
				out.add(std::to_string(var.v));
		},
		varcase(const ExprType::NUMERAL_U64) {
			out.add(std::to_string(var.v));
		},
		varcase(const ExprType::NUMERAL_I128) {
			out.add("0x");
			writeU64Hex(out, var.hi);
			writeU64Hex(out, var.lo);
		},

		varcase(const ExprType::NUMERAL_U128) {
			out.add("0x");
			writeU64Hex(out, var.hi);
			writeU64Hex(out, var.lo);
		},

		varcase(const ExprType::LITERAL_STRING&) {
			genLiteral(out,var.v);
		},
		varcase(const ExprType::FUNCTION_DEF<Out>&) {
			genFuncDef(out, var.v,""sv);
		},
		varcase(const ExprType::FUNC_CALL<Out>&) {
			genFuncCall(out, var);
		},
		varcase(const ExprType::LIM_PREFIX_EXP<Out>&) {
			genLimPrefixExpr(out, *var);
		},
		varcase(const ExprType::TABLE_CONSTRUCTOR<Out>&) {
			genTableConstructor(out, var.v);
		},
		varcase(const ExprType::MULTI_OPERATION<Out>&) {
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
	template<AnyOutput Out>
	inline void genExpList(Out& out, const ExpList<Out>& obj)
	{
		for (const Expression<Out>& e : obj)
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

	template<AnyOutput Out>
	inline void genArgFuncCall(Out& out, const ArgFuncCall<Out>& arg)
	{
		if (!arg.funcName.empty())
		{
			out.add(':')
				.add(arg.funcName);
		}
		ezmatch(arg.args)(
		varcase(const ArgsType::EXPLIST<Out>&) {
			out.add('(');
			genExpList(out, var.v);
			out.add(')');
		},
		varcase(const ArgsType::TABLE<Out>&) {
			genTableConstructor(out, var.v);
		},
		varcase(const ArgsType::LITERAL&) {
			genLiteral(out, var.v);
		}
		);
	}

	template<AnyOutput Out>
	inline void genSubVar(Out& out, const SubVar<Out>& obj)
	{
		for (const ArgFuncCall<Out>& arg : obj.funcCalls)
		{
			genArgFuncCall(out, arg);
		}
		ezmatch(obj.idx)(
		varcase(const SubVarType::EXPR<Out>&) {
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

	template<AnyOutput Out>
	inline void genVar(Out& out, const Var<Out>& obj)
	{
		ezmatch(obj.base)(
		varcase(const BaseVarType::NAME&) {
			out.add(var);
		},
		varcase(const BaseVarType::MOD_PATH&) {
			out.add(var[0]);
			for (size_t i = 1; i < var.size(); i++)
			{
				out.add("::");
				out.add(var[i]);
			}
		},
		varcase(const BaseVarType::EXPR<Out>&) {
			out.add('(');
			genExpr(out, var.start);
			out.add(')');
			genSubVar(out, var.sub);
		}
		);
		for (const SubVar<Out>& sub :  obj.sub)
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

	template<AnyOutput Out>
	inline void genVarList(Out& out, const std::vector<Var<Out>>& obj)
	{
		for (const Var<Out>& v : obj)
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

	template<AnyOutput Out>
	inline void genStat(Out& out, const Statement<Out>& obj)
	{
		ezmatch(obj.data)(

		varcase(const StatementType::SEMICOLON) {
			if(!out.wasSemicolon)
				out.add(';');
			out.wasSemicolon = true;
		},

		varcase(const StatementType::ASSIGN<Out>&) {
			genVarList(out, var.vars);
			out.add(" = ");
			genExpList(out, var.exprs);
			out.addNewl(';');
			out.wasSemicolon = true;
		},
		varcase(const StatementType::LOCAL_ASSIGN<Out>&) {
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

		varcase(const StatementType::FUNC_CALL<Out>&) {
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
		varcase(const StatementType::BLOCK<Out>&) {
			out.newLine();//Extra spacing
			out.add(sel<Out>("do","{"))
				.tabUpNewl();
			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl(sel<Out>("end", "}"));
		},
		varcase(const StatementType::WHILE_LOOP<Out>&) {
			out.newLine();//Extra spacing
			out.add("while ");

			if constexpr (out.settings() & sluaSyn)out.add('(');
			genExpr(out, var.cond);

			out.add(sel<Out>(" do", ")"));

			if constexpr (out.settings() & sluaSyn) out.newLine().add('{');
			out.tabUpNewl();

			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl(sel<Out>("end","}"));
		},
		varcase(const StatementType::REPEAT_UNTIL<Out>&) {
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

		varcase(const StatementType::IF_THEN_ELSE<Out>&) {
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

		varcase(const StatementType::FOR_LOOP_NUMERIC<Out>&) {
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
		varcase(const StatementType::FOR_LOOP_GENERIC<Out>&) {
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

		varcase(const StatementType::FUNCTION_DEF<Out>&) {
			genFuncDef(out, var.func,var.name);
		},
		varcase(const StatementType::LOCAL_FUNCTION_DEF<Out>&) {
			out.add("local ");
			genFuncDef(out, var.func, var.name);
		}

		);
	}

	template<AnyOutput Out>
	inline void genBlock(Out& out, const Block<Out>& obj)
	{
		for (const Statement<Out>& s : obj.statList)
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

	template<AnyOutput Out>
	inline void genFile(Out& out,const ParsedFile<Out>& obj)
	{
		genBlock(out, obj.code);
	}
}
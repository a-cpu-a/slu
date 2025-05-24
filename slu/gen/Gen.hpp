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

#include <slu/parser/State.hpp>
#include <slu/parser/Parse.hpp>
#include <slu/ext/CppMatch.hpp>
#include <slu/gen/Output.hpp>


namespace slu::parse
{
	template<AnyCfgable Out>
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
			return sel<Out>("..", "++");
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
			return sel<Out>("~=", "!=");
		case BinOpType::LOGICAL_AND:
			return "and"sv;
		case BinOpType::LOGICAL_OR:
			return "or"sv;
			// Slu
		case BinOpType::ARRAY_MUL:
			return "**"sv;
		case BinOpType::RANGE_BETWEEN:
			return ".."sv;
		default:
			_ASSERT(false);
			return "<ERROR>"sv;
		}
	}
	template<AnyCfgable Out>
	inline std::string_view getUnOpAsStr(const UnOpType t)
	{
		using namespace std::literals;
		switch (t)
		{
		case UnOpType::NEGATE:
			return " -"sv;//TODO: elide space, when there is one already
		case UnOpType::LOGICAL_NOT:
			return sel<Out>(" not ", "!");
		case UnOpType::LENGTH:
			return "#"sv;
		case UnOpType::BITWISE_NOT:
			return "~"sv;
			// Slu
		case UnOpType::RANGE_BEFORE:
			return ".."sv;

		case UnOpType::ALLOCATE:
			return " alloc "sv;

		case UnOpType::TO_REF_MUT:
		case UnOpType::TO_REF_CONST:
		case UnOpType::TO_REF_SHARE:
			//mut or whatever missing, as lifetimes need to be added
		case UnOpType::TO_REF:
			return "&"sv;

		case UnOpType::TO_PTR:
			return "*"sv;
		case UnOpType::TO_PTR_CONST:
			return "*const "sv;
		case UnOpType::TO_PTR_SHARE:
			return "*share "sv;
		case UnOpType::TO_PTR_MUT:
			return "*mut "sv;

		case UnOpType::MUT:
			return " mut "sv;
		default:
			_ASSERT(false);
			return "<ERROR>"sv;
		}
	}
	inline std::string_view getPostUnOpAsStr(const PostUnOpType t)
	{
		using namespace std::literals;
		switch (t)
		{
			// Slu
		case PostUnOpType::RANGE_AFTER:
			return ".. "sv;

		case PostUnOpType::DEREF:
			return ".*"sv;
		case PostUnOpType::PROPOGATE_ERR:
			return "?"sv;
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
				out.add(out.db.asSv(var.idx))
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
	inline void genExprParens(Out& out, const Expression<Out>& obj)
	{
		if constexpr (out.settings() & sluSyn) out.add('(');
		genExpr(out, obj);
		if constexpr (out.settings() & sluSyn) out.add(')');
	}

	template<AnyOutput Out>
	inline void genSafety(Out& out, const OptSafety& obj)
	{
		switch (obj)
		{
		case OptSafety::DEFAULT:
			break;
		case OptSafety::SAFE:
			out.add("safe ");
			break;
		case OptSafety::UNSAFE:
			out.add("unsafe ");
			break;
		}
	}

	template<AnyOutput Out>
	inline void genTraitExpr(Out& out, const TraitExpr& obj)
	{
		if constexpr(Out::settings() & sluSyn)
		{
			for (const TraitExprItem& i : obj.traitCombo)
			{
				ezmatch(i)(
				varcase(const TypeExprDataType::FUNC_CALL&) {
					genFuncCall(out, var);
				},
				varcase(const TypeExprDataType::LIM_PREFIX_EXP&) {
					genLimPrefixExpr(out, *var);
				}
					);
				if (&i != &obj.traitCombo.back())
					out.add(" + ");
			}
		}
	}
	template<bool elideStruct,AnyOutput Out>
	inline void genTypeExprData(Out& out, const TypeExprData& obj)
	{
		ezmatch(obj)(
		varcase(const TypeExprDataType::DYN&) {
			out.add("dyn ");
			genTraitExpr(out,var.expr);
		},
		varcase(const TypeExprDataType::IMPL&) {
			out.add("impl ");
			genTraitExpr(out,var.expr);
		},
		varcase(const TypeExprDataType::ERR&) {
			out.add("//");
			genTypeExpr(out,*var.err);
		},
		varcase(const TypeExprDataType::SLICER&) {
			out.add("[");
			genExpr(out,*var);
			out.add("]");
		},
		varcase(const TypeExprDataType::Union&) {
			out.add("union ");
			genTableConstructor(out, var.fields);
		},
		varcase(const TypeExprDataType::Struct&) {
			if constexpr(!elideStruct)
				out.add("struct ");
			genTableConstructor(out, var);
		},
		varcase(const TypeExprDataType::FN&) {
			genSafety(out, var.safety);
			out.add("fn ");
			genTypeExpr(out, *var.argType);
			out.add(" -> ");
			genTypeExpr(out, *var.retType);
		},
		varcase(const TypeExprDataType::ERR_INFERR) {
			out.add("?");
		},
		varcase(const TypeExprDataType::TRAIT_TY) {
			out.add("trait");
		},
		varcase(const TypeExprDataType::FUNC_CALL&) {
			genFuncCall(out, var);
		},
		varcase(const TypeExprDataType::LIM_PREFIX_EXP&) {
			genLimPrefixExpr(out, *var);
		},
		varcase(const TypeExprDataType::MULTI_OP&) {
			genTypeExpr(out, *var.first);
			for (const auto& [op, ex] : var.extra)
			{
				out.add(' ')
					.add(getBinOpAsStr<Out>(op))
					.add(' ');
				genTypeExpr(out, ex);
			}
		},
		varcase(const TypeExprDataType::NUMERAL_I64) {
			out.add(std::to_string(var.v));
		},
		varcase(const TypeExprDataType::NUMERAL_U64) {
			out.add(std::to_string(var.v));
		},
		varcase(const TypeExprDataType::NUMERAL_I128) {
			out.add(parse::u128ToStr(var.lo, var.hi));
		},
		varcase(const TypeExprDataType::NUMERAL_U128) {
			out.add(parse::u128ToStr(var.lo, var.hi));
		}
		);
	}

	template<bool elideStruct=false, AnyOutput Out>
	inline void genTypeExpr(Out& out, const TypeExpr& obj)
	{
		if constexpr (Out::settings() & sluSyn)
		{
			if (obj.hasMut)
				out.add("mut ");
			genUnOps(out, obj.unOps);
			genTypeExprData<elideStruct>(out, obj.data);
			for (const PostUnOpType t : obj.postUnOps)
				out.add(getPostUnOpAsStr(t));
		}
	}

	template<AnyOutput Out>
	inline void genLifetime(Out& out, const Lifetime& obj)
	{
		for (MpItmId<Out> i : obj)
			out.add('/').add(out.db.asSv(i));
	}

	template<AnyOutput Out>
	inline void genUnOps(Out& out, const auto& obj)
	{
		for (const UnOpItem& t : obj)
		{
			out.add(getUnOpAsStr<Out>(t.type));
			if constexpr (out.settings() & sluSyn)
			{
				if (t.type == UnOpType::TO_REF
					|| t.type == UnOpType::TO_REF_MUT
					|| t.type == UnOpType::TO_REF_CONST
					|| t.type == UnOpType::TO_REF_SHARE)
				{
					genLifetime(out, t.life);
					if (!t.life.empty())
						out.add(' ');

					if (t.type == UnOpType::TO_REF_MUT)
						out.add("mut ");
					else if (t.type == UnOpType::TO_REF_CONST)
						out.add("const ");
					else if (t.type == UnOpType::TO_REF_SHARE)
						out.add("share ");
				}
			}
		}
	}
	template<AnyOutput Out>
	inline void genExpr(Out& out, const Expression<Out>& obj)
	{
		genUnOps(out,obj.unOps);

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
			_ASSERT(!(Out::settings() & parse::noIntOverflow) || var.v >= 0);
			if (var.v < 0)
			{
				out.add("0x");
				writeU64Hex(out,var.v);
			}
			else
				out.add(std::to_string(var.v));
		},

		varcase(const ExprType::LITERAL_STRING&) {
			genLiteral(out,var.v);
		},
		varcase(const ExprType::FUNCTION_DEF<Out>&) {
			out.add("function ");
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
					.add(getBinOpAsStr<Out>(op))
					.add(' ');
				genExpr(out, ex);
			}
		},
		varcase(const ExprType::OPEN_RANGE) {
			out.add("..");
		},
		varcase(const ExprType::LIFETIME&) {
			if constexpr (out.settings() & sluSyn)
				genLifetime(out, var);
		},
		varcase(const ExprType::TYPE_EXPR&) {
			genTypeExpr(out, var);
		},
		varcase(const ExprType::TRAIT_EXPR&) {
			genTraitExpr(out, var);
		},
		varcase(const ExprType::IfCond<Out>&) {
			genIfCond<true>(out, var);
		},
		varcase(const ExprType::PAT_TYPE_PREFIX&) {},//Yes, nothing
		varcase(const ExprType::NUMERAL_U64) {
			out.add(std::to_string(var.v));
		},
		varcase(const ExprType::NUMERAL_I128) {
			out.add(parse::u128ToStr(var.lo, var.hi));
		},
		varcase(const ExprType::NUMERAL_U128) {
			out.add(parse::u128ToStr(var.lo, var.hi));
		}
		);
		if constexpr(out.settings()&sluSyn)
		{
			for (const PostUnOpType t : obj.postUnOps)
				out.add(getPostUnOpAsStr(t));
		}
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
				.add(out.db.asSv(arg.funcName));
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
			out.add(' '); genLiteral(out, var.v);
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
		varcase(const SubVarType::NAME<Out>&) {
			const std::string_view txt = out.db.asSv(var.idx);
			if (txt.empty())return;
			out.add('.')
				.add(txt);
		},
		varcase(const SubVarType::DEREF) {
			out.add(".*");
		}
		);
	}

	template<AnyOutput Out>
	inline void genModPath(Out& out, const lang::ViewModPath& obj)
	{
		out.add(obj[0]);
		for (size_t i = 1; i < obj.size(); i++)
		{
			out.add("::");
			out.add(obj[i]);
		}
	}
	template<AnyOutput Out>
	inline void genVar(Out& out, const Var<Out>& obj)
	{
		ezmatch(obj.base)(
		varcase(const BaseVarType::NAME<Out>&) {
			out.add(out.db.asSv(var.v));
		},
		varcase(const BaseVarType::EXPR<Out>&) {
			out.add('(');
			genExpr(out, var.start);
			out.add(')');
		}
		);
		for (const SubVar<Out>& sub :  obj.sub)
		{
			genSubVar(out, sub);
		}
	}
	template<AnyOutput Out>
	inline void genParamList(Out& out, const ParamList<Out>& itm,const bool hasVarArgParam)
	{
		for (const Parameter<Out>& par : itm)
		{
			if constexpr (out.settings() & sluSyn)
				genPat(out, par.name);
			else
				out.add(out.db.asSv(par.name));

			if (&par != &itm.back() || hasVarArgParam)
				out.add(", ");
		}
		if (hasVarArgParam)
			out.add("...");
	}
	template<AnyOutput Out>
	inline void genFuncDef(Out& out, const Function<Out>& var,const std::string_view name)
	{
		out.add(name)
			.add('(');

		genParamList(out, var.params,var.hasVarArgParam);

		out.add(')');

		if constexpr (out.settings() & sluSyn)
			out.newLine().add('{');
		out.tabUpNewl();

		genBlock(out, var.block);

		out.unTabNewl()
			.addNewl(sel<Out>("end","}"));

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
	template<AnyOutput Out>
	inline void genDestrSpec(Out& out, const DestrSpec& obj)
	{
		ezmatch(obj)(
		varcase(const DestrSpecType::Type&) {
			out.add("as ");
			genTypeExpr<true>(out, var);
			out.add(' ');
		},
		varcase(const DestrSpecType::Spat&) {
			genExpr(out, var);
			out.add(' ');
		},
		varcase(const DestrSpecType::Prefix&) {
			genUnOps(out, var);
		}
		);
	}
	template<AnyOutput Out>
	inline void genPat(Out& out, const Pat& obj)
	{
		ezmatch(obj)(
		varcase(const PatType::DestrAny) {
			out.add('_');
		},
		varcase(const PatType::Simple&) {
			genExpr(out, var);
		},
			// Fields / List
		varcase(const AnyCompoundDestr auto&) {
			genDestrSpec(out, var.spec);
			out.add('{').tabUpNewl();

			constexpr bool isList = std::is_same_v<decltype(var), const PatType::DestrList&>;

			for (const auto& field : var.items)
			{
				if constexpr(isList)
					genPat(out, field);
				else
				{
					out.add('|')
						.add(out.db.asSv(field.name))
						.add("| ");

					genPat(out, field.pat);
				}
				if (&field != &var.items.back())
					out.add(",").newLine();
			}

			if(var.extraFields)
				out.add(", ..");
			out.unTabNewl().add('}');
			if(!var.name.empty())
				out.add(' ').add(out.db.asSv(var.name));
		},
		varcase(const PatType::DestrName&) {
			genDestrSpec(out, var.spec);
			out.add(out.db.asSv(var.name));
		},
		varcase(const PatType::DestrNameRestrict&) {
			genDestrSpec(out, var.spec);
			out.add(out.db.asSv(var.name));
			out.add(" = ");
			genExpr(out, var.restriction);
		}
		);
	}
	template<AnyOutput Out>
	inline void genAtribNameList(Out& out, const AttribNameList<Out>& obj)
	{
		for (const AttribName<Out>& v : obj)
		{
			out.add(out.db.asSv(v.name));
			if (!v.attrib.empty())
				out.add(" <")
				.add(v.attrib)
				.add('>');
			if (&v != &obj.back())
				out.add(", ");
		}
	}
	template<AnyOutput Out>
	inline void genNames(Out& out, const NameList<Out>& obj)
	{
		for (const MpItmId<Out>& v : obj)
		{
			out.add(out.db.asSv(v));
			if (&v != &obj.back())
				out.add(", ");
		}
	}
	inline void genUseVariant(AnyOutput auto& out, const UseVariant& obj)
	{
		ezmatch(obj)(
		varcase(const UseVariantType::EVERYTHING_INSIDE&){
			out.add("::*");
		},
		varcase(const UseVariantType::AS_NAME&){
			out.add(" as ").add(out.db.asSv(var));
		},
		varcase(const UseVariantType::IMPORT&){},
		varcase(const UseVariantType::LIST_OF_STUFF&)
		{
			out.add("::{").add(out.db.asSv(var[0]));
			for (size_t i = 1; i < var.size(); i++)
			{
				out.add(", ").add(out.db.asSv(var[i]));
			}
			out.add("}");
		}
		);
	}

	template<AnyOutput Out>
	inline void genStatOrExpr(Out& out, const parse::StatOrExpr<Out>& obj)

	{
		ezmatch(obj)(
		varcase(const parse::StatOrExprType::BLOCK<Out>&) {
			out.newLine().add('{').tabUpNewl();
			genBlock(out, var);
			out.unTabNewl().add('}');
		},
		varcase(const parse::StatOrExprType::EXPR<Out>&) {
			out.add(" => ");
			genExpr(out, var);
		}
		);
	}
	template<bool isExpr,AnyOutput Out>
	inline void genIfCond(Out& out,
		const parse::BaseIfCond<Out, isExpr>& itm)
	{
		out.add("if ");

		if constexpr (Out::settings() & sluSyn)
		{
			genExpr(out, *itm.cond);
			genStatOrExpr(out, *itm.bl);
			if constexpr (isExpr)out.add(' ');
		}
		else
		{
			genExprParens(out, *itm.cond);
			out.add(" then").tabUpNewl();
			genBlock(out, *itm.bl);
		}

		if (!itm.elseIfs.empty())
		{
			for (const auto& [expr, bl] : itm.elseIfs)
			{
				if constexpr (!(Out::settings() & sluSyn))
					out.unTabNewl();
				out.add(sel<Out>("elseif ", "else if "));

				if constexpr (Out::settings() & sluSyn)
				{
					genExpr(out, expr);
					genStatOrExpr(out, bl);
					if constexpr (isExpr)out.add(' ');
				}
				else
				{
					genExprParens(out, expr);
					out.add(" then").tabUpNewl();
					genBlock(out, bl);
				}
			}
		}
		if (itm.elseBlock)
		{
			if constexpr (!(Out::settings() & sluSyn))
				out.unTabNewl();
			out.add("else");

			if constexpr (Out::settings() & sluSyn)
			{
				genStatOrExpr(out, **itm.elseBlock);
			}
			else
			{
				out.tabUpNewl();
				genBlock(out, **itm.elseBlock);
			}
		}

		if constexpr (!(Out::settings() & sluSyn))
			out.unTabNewl().addNewl("end");
	}

	template<size_t N,AnyOutput Out>
	inline void genVarStat(Out& out, const auto& obj,const char(&kw)[N])
	{
		if constexpr (Out::settings() & sluSyn)
		{
			if (obj.exported)
				out.add("ex ");
		}

		out.add(kw);
		if constexpr (Out::settings() & sluSyn)
			genPat(out, obj.names);
		else
			genAtribNameList(out, obj.names);
		if (!obj.exprs.empty())
		{
			out.add(" = ");
			genExpList(out, obj.exprs);
		}
		out.addNewl(';');
		out.wasSemicolon = true;
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
			genVarStat(out, var,"local ");
		},
		varcase(const StatementType::LET<Out>&) {
			genVarStat(out, var, "let ");
		},
		varcase(const StatementType::CONST<Out>&) {
			genVarStat(out, var, "const ");
		},

		varcase(const StatementType::FUNC_CALL<Out>&) {
			genFuncCall(out, var);
			out.addNewl(';');
			out.wasSemicolon = true;
		},
		varcase(const StatementType::LABEL<Out>&) {
			out.unTabTemp()
				.add(sel<Out>("::", ":::"))
				.add(out.db.asSv(var.v))
				.addNewl(sel<Out>("::", ":"))
				.tabUpTemp();
		},
		varcase(const StatementType::BREAK) {
			out.addNewl("break;");
			out.wasSemicolon = true;
		},
		varcase(const StatementType::GOTO<Out>&) {
			out.add("goto ")
				.add(out.db.asSv(var.v))
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
		varcase(const StatementType::IfCond<Out>&) {
			genIfCond<false>(out, var);
		},
		varcase(const StatementType::WHILE_LOOP<Out>&) {
			out.newLine();//Extra spacing
			out.add("while ");

			genExprParens(out, var.cond);

			if constexpr (Out::settings() & sluSyn) 
				out.newLine().add('{');
			else
				out.add(" do");

			out.tabUpNewl();

			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl(sel<Out>("end","}"));
		},
		varcase(const StatementType::REPEAT_UNTIL<Out>&) {
			out.add("repeat");

			if constexpr (Out::settings() & sluSyn)
				out.newLine().add('{');
			out.tabUpNewl();

			genBlock(out, var.bl);
			out.unTabNewl();

			if constexpr (Out::settings() & sluSyn)
				out.add('}');
			out.add("until ");
			genExpr(out, var.cond);
			out.addNewl(';');

			out.newLine();//Extra spacing
			out.wasSemicolon = true;
		},
		varcase(const StatementType::FOR_LOOP_NUMERIC<Out>&) {

			out.add("for ");
			if constexpr (Out::settings() & sluSyn)
				genPat(out, var.varName);
			else
				out.add(out.db.asSv(var.varName));
			out.add(" = ");
			genExpr(out, var.start);
			out.add(", ");
			genExpr(out, var.end);
			if (var.step)
			{
				out.add(", ");
				genExpr(out, *var.step);
			}
			if constexpr (Out::settings() & sluSyn)
				out.newLine().add('{');
			else
				out.add(" do");

			out.tabUpNewl();

			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl(sel<Out>("end", "}"));
		},
		varcase(const StatementType::FOR_LOOP_GENERIC<Out>&) {
			out.add("for ");
			if constexpr (Out::settings() & sluSyn)
				genPat(out, var.varNames);
			else
				genNames(out, var.varNames);
			out.add(" in ");
			if constexpr (Out::settings() & sluSyn)
				genExpr(out, var.exprs);
			else
				genExpList(out, var.exprs);

			if constexpr (Out::settings() & sluSyn)
				out.newLine().add('{');
			else
				out.add(" do");
			out.tabUpNewl();

			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl(sel<Out>("end", "}"));
		},

		varcase(const StatementType::FN<Out>&) {
			if constexpr (Out::settings() & sluSyn)
			{
				if (var.exported)out.add("ex ");
				genSafety(out, var.func.safety);
			}
			out.add("fn ");
			genFuncDef(out, var.func, out.db.asSv(var.name));
		},

		varcase(const StatementType::FUNCTION_DEF<Out>&) {
			if constexpr (Out::settings() & sluSyn)
			{
				if (var.exported)out.add("ex ");
				genSafety(out, var.func.safety);
			}
			out.add("function ");
			genFuncDef(out, var.func, out.db.asSv(var.name));
		},
		varcase(const StatementType::LOCAL_FUNCTION_DEF<Out>&) {
			out.add("local function ");
			genFuncDef(out, var.func, out.db.asSv(var.name));
		},

		//Slu!

		varcase(const StatementType::Struct<Out>&) {
			if (var.exported)out.add("ex ");
			out.add("struct ").add(out.db.asSv(var.name));
			if (!var.params.empty())
			{
				out.add('(');
				genParamList(out, var.params,false);
				out.add(')');
			}
			if (!var.type.isBasicStruct())
				out.add(" = ");
			else
				out.add(" ");

			genTypeExpr<true>(out, var.type);
			out.newLine();
		},

		varcase(const StatementType::Union<Out>&) {
			if (var.exported)out.add("ex ");
			out.add("union ").add(out.db.asSv(var.name));
			if (!var.params.empty())
			{
				out.add('(');
				genParamList(out, var.params,false);
				out.add(')');
			}
			out.add(' ');
			genTableConstructor(out, var.type);
			out.newLine();
		},

		varcase(const StatementType::UnsafeBlock<Out>&) {
			out.newLine();//Extra spacing
			out.add("unsafe {")
				.tabUpNewl();
			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl("}");
		},
		varcase(const StatementType::UNSAFE_LABEL) {
			out.unTabTemp()
				.add(":::unsafe:")
				.tabUpTemp();
		},
		varcase(const StatementType::SAFE_LABEL) {
			out.unTabTemp()
				.add(":::safe:")
				.tabUpTemp();
		},

		varcase(const StatementType::DROP<Out>&) {
			out.add("drop ");;
			genExpr(out, var.expr); 
			out.addNewl(";");
			out.wasSemicolon = true;
		},
		varcase(const StatementType::USE&) {
			if (var.exported)out.add("ex ");
			out.add("use ");
			genModPath(out, out.db.asVmp(var.base));
			genUseVariant(out, var.useVariant);
			out.addNewl(";");
			out.wasSemicolon = true;
		},
		varcase(const StatementType::MOD_DEF<Out>&) {
			if (var.exported)out.add("ex ");
			out.add("mod ").add(out.db.asSv(var.name)).addNewl(";");
			out.wasSemicolon = true;
		},
		varcase(const StatementType::MOD_DEF_INLINE<Out>&) {
			if (var.exported)out.add("ex ");
			out.add("mod ").add(out.db.asSv(var.name)).add(" as {");
			out.tabUpNewl().newLine();

			genBlock(out,var.bl);
			out.unTabNewl().add("}");
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
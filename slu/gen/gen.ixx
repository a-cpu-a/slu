/*
    Slu language compiler, a computer program compiler.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>

#include <slu/ext/CppMatch.hpp>
#include <slu/Panic.hpp>
export module slu.gen.gen;

import slu.char_info;
import slu.num;
import slu.settings;
import slu.ast.enums;
import slu.ast.state;
import slu.ast.state_decls;
import slu.gen.output;
import slu.lang.basic_state;
import slu.parse.parse;

namespace slu::parse
{
	template<AnyOutput Out> inline void genStat(Out& out, const Stat& obj);
	template<AnyOutput Out> inline void genExpr(Out& out, const Expr& obj);

	template<AnyOutput Out>
	inline void genExprList(Out& out, const ExprList& obj)
	{
		for (const Expr& e : obj)
		{
			genExpr(out, e);
			if (&e != &obj.back())
				out.add(", ");
		}
	}
	template<AnyOutput Out>
	inline void genBlock(Out& out, const Block<Out>& itm)
	{
		for (const Stat& s : itm.statList)
			genStat(out, s);

		if (itm.retTy != parse::RetType::NONE)
		{
			switch (itm.retTy)
			{
			case parse::RetType::RETURN:   out.add("return"); break;
			case parse::RetType::BREAK:    out.add("break"); break;
			case parse::RetType::CONTINUE: out.add("continue"); break;
			case parse::RetType::NONE:
				Slu_panic("RetType was NONE, even after checking that it isnt");
			}
			if (!itm.retExprs.empty())
			{
				out.add(' ');
				genExprList(out, itm.retExprs);
			}
			out.add(';');
			out.wasSemicolon = true;
		}
	}

	template<AnyCfgable Out>
	inline std::string_view getBinOpAsStr(const ast::BinOpType t)
	{
		using namespace std::literals;
		switch (t)
		{
		case ast::BinOpType::ADD:       return "+"sv;
		case ast::BinOpType::SUB:       return "-"sv;
		case ast::BinOpType::MUL:       return "*"sv;
		case ast::BinOpType::DIV:       return "/"sv;
		case ast::BinOpType::FLOOR_DIV: return "//"sv;
		case ast::BinOpType::EXP:       return "^"sv;
		case ast::BinOpType::REM:       return "%"sv;

		case ast::BinOpType::BIT_AND: return "&"sv;
		case ast::BinOpType::BIT_XOR: return "~"sv;
		case ast::BinOpType::BIT_OR:  return "|"sv;
		case ast::BinOpType::SHR:     return ">>"sv;
		case ast::BinOpType::SHL:     return "<<"sv;

		case ast::BinOpType::LT: return "<"sv;
		case ast::BinOpType::LE: return "<="sv;
		case ast::BinOpType::GT: return ">"sv;
		case ast::BinOpType::GE: return ">="sv;

		case ast::BinOpType::EQ: return "=="sv;
		case ast::BinOpType::NE: return "!="sv;

		case ast::BinOpType::AND: return "and"sv;
		case ast::BinOpType::OR:  return "or"sv;

		case ast::BinOpType::CONCAT:    return "++"sv;
		case ast::BinOpType::REP:       return "**"sv;
		case ast::BinOpType::RANGE:     return ".."sv;
		case ast::BinOpType::MK_RESULT: return "~~"sv;
		case ast::BinOpType::UNION:     return "||"sv;

		case ast::BinOpType::AS: return "as"sv;
		default:                 Slu_panic("Unknown binop");
		}
	}
	template<AnyCfgable Out>
	inline std::string_view getUnOpAsStr(const ast::UnOpType t)
	{
		using namespace std::literals;
		switch (t)
		{
		case ast::UnOpType::NEG:
			return " -"sv; //TODO: elide space, when there is one already
		case ast::UnOpType::NOT: return "!"sv;

		case ast::UnOpType::RANGE_BEFORE: return ".."sv;
		case ast::UnOpType::ALLOC:        return " alloc "sv;

		case ast::UnOpType::REF_MUT:
		case ast::UnOpType::REF_CONST:
		case ast::UnOpType::REF_SHARE:
			//mut or whatever missing, as lifetimes need to be added
		case ast::UnOpType::REF: return "&"sv;

		case ast::UnOpType::PTR:       return "*"sv;
		case ast::UnOpType::PTR_CONST: return "*const "sv;
		case ast::UnOpType::PTR_SHARE: return "*share "sv;
		case ast::UnOpType::PTR_MUT:   return "*mut "sv;
		case ast::UnOpType::SLICIFY:   return "[]"sv;

		case ast::UnOpType::MARK_MUT: return " mut "sv;
		default:                      Slu_panic("Unknown unop");
		}
	}
	inline std::string_view getPostUnOpAsStr(const ast::PostUnOpType t)
	{
		using namespace std::literals;
		switch (t)
		{
			// Slu
		case ast::PostUnOpType::RANGE_AFTER: return ".. "sv;

		case ast::PostUnOpType::DEREF: return ".*"sv;
		case ast::PostUnOpType::TRY:   return "?"sv;
		default:                       Slu_panic("Unknown post unop");
		}
	}

	template<AnyOutput Out>
	inline void genTable(Out& out, const Table<Out>& obj)
	{
		out.add('{').template tabUpNewl<false>();

		for (const Field<Out>& f : obj)
		{
			ezmatch(f)(
			    varcase(const FieldType::NONE) {
				    Slu_panic("Table field was FieldType::NONE");
			    },

			    varcase(const FieldType::Expr2Expr&) {
				    out.addIndent();
				    out.add('[');
				    genExpr(out, var.idx);
				    out.add("] = ");
				    genExpr(out, var.v);
			    },
			    varcase(const FieldType::Name2Expr&) {
				    out.addIndent();
				    out.add(out.db.asSv(var.idx)).add(" = ");
				    genExpr(out, var.v);
			    },
			    varcase(const FieldType::Expr&) {
				    out.addIndent();
				    genExpr(out, var);
			    });
			out.template addNewl<false>(',');
		}

		out.unTab().add('}');
	}
	inline void writeU64Hex(AnyOutput auto& out, const uint64_t v)
	{
		for (size_t i = 0; i < 16; i++)
		{
			const uint64_t val = (uint64_t(v) >> (60 - 4 * i));
			if (val == 0)
				continue;
			const uint8_t c = val & 0xF;
			out.add(numToHex(c));
		}
		if (v == 0)
			out.add('0');
	}
	inline void genString(AnyOutput auto& out, const std::string& obj)
	{
		out.add('"');
		for (const char ch : obj)
		{
			switch (ch)
			{
			case '\n': out.add("\\n"); break;
			case '\r': out.add("\\r"); break;
			case '\t': out.add("\\t"); break;
			case '\b': out.add("\\b"); break;
			case '\a': out.add("\\a"); break;
			case '\f': out.add("\\f"); break;
			case '\v': out.add("\\v"); break;
			case '"':  out.add("\\\""); break;
			case '\\': out.add("\\\\"); break;
			case '\0': out.add("\\x00"); break;
			default:   out.add(ch); break;
			}
		}
		out.add('"');
	}
	template<bool isLocal, AnyOutput Out>
	inline void genNameOrLocal(Out& out, const LocalOrName<Out, isLocal>& obj)
	{
		std::string_view v;
		if constexpr (isLocal)
			v = (out.db.asSv(out.resolveLocal(obj)));
		else
			v = (out.db.asSv(obj));
		if (v.starts_with('$'))
		{
			out.add("--[=[ $0x");
			uint64_t synId = (uint64_t(v[1]) << 56) | (uint64_t(v[2]) << 48)
			    | (uint64_t(v[3]) << 40) | (uint64_t(v[4]) << 32)
			    | (uint64_t(v[5]) << 24) | (uint64_t(v[6]) << 16)
			    | (uint64_t(v[7]) << 8) | v[8];
			writeU64Hex(out, synId);
			out.add(" ]=]");
			return;
		}
		out.add(v);
	}

	template<AnyOutput Out> inline void genExprParens(Out& out, const Expr& obj)
	{
		out.add('(');
		genExpr(out, obj);
		out.add(')');
	}

	template<AnyOutput Out>
	inline void genSafety(Out& out, const ast::OptSafety& obj)
	{
		switch (obj)
		{
		case ast::OptSafety::DEFAULT: break;
		case ast::OptSafety::SAFE:    out.add("safe "); break;
		case ast::OptSafety::UNSAFE:  out.add("unsafe "); break;
		}
	}

	template<AnyOutput Out>
	inline void genExSafety(
	    Out& out, const bool exported, const ast::OptSafety& safety)
	{
		if (exported)
			out.add("ex ");
		genSafety(out, safety);
	}

	template<AnyOutput Out>
	inline void genTraitExpr(Out& out, const TraitExpr& obj)
	{
		for (const parse::Expr& i : obj.traitCombo)
		{
			genExpr(out, i);
			if (&i != &obj.traitCombo.back())
				out.add(" + ");
		}
	}

	template<AnyOutput Out>
	inline void genLifetime(Out& out, const Lifetime& obj)
	{
		for (lang::MpItmId i : obj)
			out.add('/').add(out.db.asSv(i));
	}

	template<AnyOutput Out>
	inline void genParamList(
	    Out& out, const ParamList& itm, const bool hasVarArgParam)
	{
		for (const Parameter& par : itm)
		{
			ezmatch(par.name)(
			    varcase(
			        const parse::LocalId&) { genNameOrLocal<true>(out, var); },
			    varcase(const lang::MpItmId&) {
				    out.add("const ");
				    genNameOrLocal<false>(out, var);
			    });
			out.add(" = ");
			genExpr(out, par.type);

			if (&par != &itm.back() || hasVarArgParam)
				out.add(", ");
		}
		if (hasVarArgParam)
			out.add("...");
	}
	template<AnyOutput Out>
	inline void genFuncDecl(
	    Out& out, const auto& itm, const std::string_view name)
	{
		out.add(name);
		out.add('(');
		if (!itm.selfArg.empty())
		{
			for (const ast::UnOpType t : itm.selfArg.specifiers)
				out.add(getUnOpAsStr<Out>(t));
			out.add("self");
			if (!itm.params.empty())
				out.add(", ");
		}
		genParamList(out, itm.params, itm.hasVarArgParam);
		out.add(')');

		if (itm.retType.has_value())
		{
			out.add(" -> ");
			genExpr(out, **itm.retType);
		}
	}
	template<AnyOutput Out>
	inline void genFuncDef(
	    Out& out, const Function& var, const std::string_view name)
	{
		out.pushLocals(var.local2Mp);
		genFuncDecl(out, var, name);

		out.newLine().add('{').tabUpNewl();
		genBlock(out, var.block);
		out.unTabNewl().addNewl("}").newLine(); //Extra spacing

		out.popLocals();
	}
	template<bool isDecl, AnyOutput Out>
	inline void genFunc(Out& out, const auto& itm, const std::string_view kw)
	{
		if constexpr (isDecl)
		{
			genExSafety(out, itm.exported, itm.safety);
			out.pushLocals(itm.local2Mp);

			out.add(kw);
			genFuncDecl(out, itm, out.db.asSv(itm.name));
			out.addNewl(";");
			out.wasSemicolon = true;
			out.popLocals();
		} else
		{
			genExSafety(out, itm.exported, itm.func.safety);

			out.add(kw);
			genFuncDef(out, itm.func, out.db.asSv(itm.name));
		}
	}

	template<AnyOutput Out> inline void genArgs(Out& out, const Args& itm)
	{
		ezmatch(itm)(
		    varcase(const ArgsType::ExprList&) {
			    out.add('(');
			    genExprList(out, var);
			    out.add(')');
		    },
		    varcase(const ArgsType::Table<Out>&) { genTable(out, var); },
		    varcase(const ArgsType::String&) {
			    out.add(' ');
			    genString(out, var.v);
		    });
	}

	template<bool boxed, AnyOutput Out>
	inline void genCall(Out& out, const Call<boxed>& itm)
	{
		genExpr(out, *itm.v);
		genArgs(out, itm.args);
	}
	template<bool boxed, AnyOutput Out>
	inline void genSelfCall(Out& out, const SelfCall<boxed>& itm)
	{
		genExpr(out, *itm.v);
		out.add('.').add(out.db.asSv(itm.method));
		genArgs(out, itm.args);
	}

	template<AnyOutput Out>
	inline void genModPath(Out& out, const lang::ViewModPath& obj)
	{
		bool unresolved = obj[0].empty();
		if (!unresolved)
		{
			out.add(":>::");
			out.add(obj[0]);
		}
		for (size_t i = 1; i < obj.size(); i++)
		{
			if (!unresolved || i != 1)
				out.add("::");
			out.add(obj[i]);
		}
	}

	template<AnyOutput Out>
	inline void genSoe(
	    Out& out, const parse::Soe<Out>& obj, const bool addArrow)
	{
		ezmatch(obj)(
		    varcase(const parse::SoeType::Block<Out>&) {
			    out.newLine().add('{').tabUpNewl();
			    genBlock(out, var);
			    out.unTabNewl().add('}');
		    },
		    varcase(const parse::SoeType::Expr&) {
			    if (addArrow)
				    out.add(" => ");
			    else
				    out.add(' ');
			    genExpr(out, *var);
		    });
	}
	template<bool isExpr, AnyOutput Out>
	inline void genIfCond(Out& out, const parse::BaseIfCond<Out, isExpr>& itm)
	{
		out.add("if ");

		genExpr(out, *itm.cond);
		genSoe(out, *itm.bl, true);
		if (isExpr)
			out.add(' ');

		for (const auto& [expr, bl] : itm.elseIfs)
		{
			out.add("else if ");

			genExpr(out, expr);
			genSoe(out, bl, true);
			if (isExpr)
				out.add(' ');
		}
		if (itm.elseBlock)
		{
			out.add("else");
			genSoe(out, **itm.elseBlock, false);
		}
	}
	template<AnyOutput Out> inline void genUnOps(Out& out, const auto& obj)
	{
		for (const UnOpItem& t : obj)
		{
			out.add(getUnOpAsStr<Out>(t.type));
			if (t.type == ast::UnOpType::REF || t.type == ast::UnOpType::REF_MUT
			    || t.type == ast::UnOpType::REF_CONST
			    || t.type == ast::UnOpType::REF_SHARE)
			{
				genLifetime(out, t.life);
				if (!t.life.empty())
					out.add(' ');

				if (t.type == ast::UnOpType::REF_MUT)
					out.add("mut ");
				else if (t.type == ast::UnOpType::REF_CONST)
					out.add("const ");
				else if (t.type == ast::UnOpType::REF_SHARE)
					out.add("share ");
			}
		}
	}
	template<AnyOutput Out>
	inline void genExprData(Out& out, const ExprData<Out>& obj)
	{
		using namespace std::literals;
		ezmatch(obj)(
		    varcase(const ExprType::Parens<Out>&) {
			    out.add('(');
			    genExpr(out, *var);
			    out.add(')');
		    },
		    varcase(const ExprType::MpRoot) { out.add(":>"sv); },
		    varcase(const ExprType::Global<Out>) {
			    genModPath(out, out.db.asVmp(var));
		    },
		    varcase(const ExprType::Local) { genNameOrLocal<true>(out, var); },
		    varcase(const ExprType::Deref&) {
			    genExpr(out, *var.v);
			    out.add(".*"sv);
		    },
		    varcase(const ExprType::Index&) {
			    genExpr(out, *var.v);
			    out.add('[');
			    genExpr(out, *var.idx);
			    out.add(']');
		    },
		    varcase(const ExprType::Field<Out>&) {
			    genExpr(out, *var.v);
			    out.add('.');
			    out.add(out.db.asSv(var.field));
		    },
		    varcase(const ExprType::Call&) { genCall<true>(out, var); },
		    varcase(const ExprType::SelfCall&) { genSelfCall<true>(out, var); },
		    varcase(const ExprType::Unfinished&) {
			    out.add("TO"
			            "DO! ");
			    genString(out, var.msg);
		    },
		    varcase(const ExprType::CompBuiltin&) {
			    out.add("_COMP_TO"
			            "DO!");
			    out.add('(');
			    genString(out, var.kind);
			    out.add(", ");
			    genExpr(out, *var.value);
			    out.add(')');
		    },
		    varcase(const ExprType::Nil) { out.add("nil"sv); },
		    varcase(const ExprType::VarArgs) { out.add("..."sv); },
		    varcase(const ExprType::F64) {
			    if (std::isinf(var) && var > 0.0f)
				    out.add("1e999"sv);
			    else
				    out.add(std::to_string(var));
		    },

		    varcase(const ExprType::I64) {
			    Slu_assert(
			        !(Out::settings() & parse::noIntOverflow) || var >= 0);
			    if (var < 0)
			    {
				    out.add("0x"sv);
				    writeU64Hex(out, var);
			    } else
				    out.add(std::to_string(var));
		    },

		    varcase(const ExprType::String&) { genString(out, var.v); },
		    varcase(const ExprType::Function&) {
			    out.add("function "sv);
			    genFuncDef(out, var, ""sv);
		    },
		    varcase(const ExprType::Table<Out>&) { genTable(out, var); },
		    varcase(const ExprType::MultiOp<Out>&) {
			    genExpr(out, *var.first);
			    for (const auto& [op, ex] : var.extra)
			    {
				    out.add(' ').add(getBinOpAsStr<Out>(op)).add(' ');
				    genExpr(out, ex);
			    }
		    },
		    varcase(const ExprType::OpenRange) { out.add(".."sv); },
		    varcase(const ExprType::Lifetime&) { genLifetime(out, var); },
		    varcase(const ExprType::TraitExpr&) { genTraitExpr(out, var); },
		    varcase(
		        const ExprType::IfCond<Out>&) { genIfCond<true>(out, var); },
		    varcase(const ExprType::PatTypePrefix&){}, //Yes, nothing
		    varcase(const ExprType::U64) { out.add(std::to_string(var)); },
		    varcase(const ExprType::P128) {
			    out.add(slu::u128ToStr(var.lo, var.hi));
		    },
		    varcase(const ExprType::M128) {
			    out.add(" -"sv).add(slu::u128ToStr(var.lo, var.hi));
		    },

		    varcase(const ExprType::Infer) { out.add('?'); },
		    varcase(const ExprType::Dyn&) {
			    out.add("dyn "sv);
			    genTraitExpr(out, var.expr);
		    },
		    varcase(const ExprType::Impl&) {
			    out.add("impl "sv);
			    genTraitExpr(out, var.expr);
		    },
		    varcase(const ExprType::Err&) {
			    out.add("~~"sv);
			    genExpr(out, *var.err);
		    },
		    varcase(const ExprType::Struct&) {
			    out.add("struct "sv);
			    genTable(out, var.fields);
		    },
		    varcase(const ExprType::Union&) {
			    out.add("union "sv);
			    genTable(out, var.fields);
		    },
		    varcase(const ExprType::FnType&) {
			    genSafety(out, var.safety);
			    out.add("fn "sv);
			    genExpr(out, *var.argType);
			    out.add(" -> "sv);
			    genExpr(out, *var.retType);
		    });
	}
	template<AnyOutput Out> inline void genExpr(Out& out, const Expr& obj)
	{
		genUnOps(out, obj.unOps);
		genExprData(out, obj.data);
		for (const ast::PostUnOpType t : obj.postUnOps)
			out.add(getPostUnOpAsStr(t));
	}

	template<AnyOutput Out>
	inline void genDestrSpec(Out& out, const DestrSpec& obj)
	{
		ezmatch(obj)(
		    varcase(const DestrSpecType::Spat&) {
			    genExpr(out, var);
			    out.add(' ');
		    },
		    varcase(const DestrSpecType::Prefix&) { genUnOps(out, var); });
	}
	template<bool isLocal, AnyOutput Out>
	inline void genPat(Out& out, const Pat<Out, isLocal>& obj)
	{
		ezmatch(obj)(
		    varcase(const PatType::DestrAny<Out, isLocal>) { out.add('_'); },
		    varcase(const PatType::Simple<Out>&) { genExpr(out, var); },
		    // Fields / List
		    [&]<class T>(const T& var)
		        requires AnyCompoundDestr<isLocal, T>
		    {
			    genDestrSpec(out, var.spec);
			    out.add('{').tabUpNewl();

			    constexpr bool isList = std::is_same_v<decltype(var),
			        const PatType::DestrList<Out, isLocal>&>;

			    for (const auto& field : var.items)
			    {
				    if constexpr (isList)
					    genPat<isLocal>(out, field);
				    else
				    {
					    out.add('|').add(out.db.asSv(field.name)).add("| ");

					    genPat<isLocal>(out, field.pat);
				    }
				    if (&field != &var.items.back())
					    out.add(',').newLine();
			    }

			    if (var.extraFields)
				    out.add(", ..");
			    out.unTabNewl().add('}');
			    out.add(' ');
			    genNameOrLocal<isLocal>(out, var.name);
		    },
		    varcase(const PatType::DestrName<Out, isLocal>&) {
			    genDestrSpec(out, var.spec);
			    genNameOrLocal<isLocal>(out, var.name);
		    },
		    varcase(const PatType::DestrNameRestrict<Out, isLocal>&) {
			    genDestrSpec(out, var.spec);
			    genNameOrLocal<isLocal>(out, var.name);
			    out.add(" = ");
			    genExpr(out, var.restriction);
		    });
	}
	template<AnyOutput Out>
	inline void genNames(Out& out, const NameList<Out>& obj)
	{
		for (const lang::MpItmId& v : obj)
		{
			out.add(out.db.asSv(v));
			if (&v != &obj.back())
				out.add(", ");
		}
	}
	inline void genUseVariant(AnyOutput auto& out, const UseVariant& obj)
	{
		ezmatch(obj)(
		    varcase(
		        const UseVariantType::EVERYTHING_INSIDE&) { out.add("::*"); },
		    varcase(const UseVariantType::AS_NAME&) {
			    out.add(" as ").add(out.db.asSv(var));
		    },
		    varcase(const UseVariantType::IMPORT&){},
		    varcase(const UseVariantType::LIST_OF_STUFF&) {
			    out.add("::{").add(out.db.asSv(var[0]));
			    for (size_t i = 1; i < var.size(); i++)
			    {
				    out.add(", ").add(out.db.asSv(var[i]));
			    }
			    out.add("}");
		    });
	}

	template<bool isLocal, size_t N, AnyOutput Out>
	inline void genVarStat(Out& out, const auto& obj, const char (&kw)[N])
	{
		genExSafety(out, obj.exported, ast::OptSafety::DEFAULT);

		out.add(kw);
		genPat<isLocal>(out, obj.names);
		if (!obj.exprs.empty())
		{
			out.add(" = ");
			genExprList(out, obj.exprs);
		}
		out.addNewl(';');
		out.wasSemicolon = true;
	}
	template<AnyOutput Out>
	inline void genWhereClauses(Out& out, const WhereClauses& itm)
	{
		if (itm.empty())
			return;
		out.add(" where ");
		for (const WhereClause& v : itm)
		{
			out.add(out.db.asSv(v.var)).add(": ");
			genTraitExpr(out, v.bound);
			if (&v != &itm.back())
				out.add(", ");
		}
	}
	template<AnyOutput Out> inline void genStat(Out& out, const Stat& obj)
	{
		ezmatch(obj.data)(

		    varcase(const StatType::Semicol) {
			    if (!out.wasSemicolon)
				    out.add(';');
			    out.wasSemicolon = true;
		    },

		    varcase(const StatType::Assign<Out>&) {
			    for (auto& i : var.vars)
				    genExprData(out, i);
			    out.add(" = ");
			    genExprList(out, var.exprs);
			    out.addNewl(';');
			    out.wasSemicolon = true;
		    },
		    varcase(const StatType::Local<Out>&) {
			    genVarStat<true>(out, var, "local ");
		    },
		    varcase(const StatType::Let<Out>&) {
			    genVarStat<true>(out, var, "let ");
		    },
		    varcase(const StatType::Const<Out>&) {
			    out.pushLocals(var.local2Mp);
			    genVarStat<false>(out, var, "const ");
			    out.popLocals();
		    },
		    varcase(const StatType::CanonicLocal&) {
			    genExSafety(out, var.exported, ast::OptSafety::DEFAULT);
			    out.add("let ");
			    //genExpr(out, var.type); //TODO: gen resolved types?
			    //out.add(' ');
			    genNameOrLocal<true>(out, var.name);
			    out.add(" = ");
			    genExpr(out, var.value);
			    out.addNewl(';');
			    out.wasSemicolon = true;
		    },
		    varcase(const StatType::CanonicGlobal&) {
			    out.pushLocals(var.local2Mp);
			    genExSafety(out, var.exported, ast::OptSafety::DEFAULT);
			    out.add("const ");
			    //genExpr(out, var.type); //TODO: gen resolved types?
			    //out.add(' ');
			    genNameOrLocal<false>(out, var.name);
			    out.add(" = ");
			    genExpr(out, var.value);
			    out.addNewl(';');
			    out.wasSemicolon = true;
			    out.popLocals();
		    },

		    varcase(const StatType::Call&) {
			    genCall<false>(out, var);
			    out.addNewl(';');
			    out.wasSemicolon = true;
		    },
		    varcase(const StatType::SelfCall&) {
			    genSelfCall<false>(out, var);
			    out.addNewl(';');
			    out.wasSemicolon = true;
		    },

		    varcase(const StatType::Label<Out>&) {
			    out.unTabTemp()
			        .add(":::")
			        .add(out.db.asSv(var.v))
			        .addNewl(':')
			        .tabUpTemp();
		    },
		    varcase(const StatType::Goto<Out>&) {
			    out.add("goto ").add(out.db.asSv(var.v)).addNewl(';');
			    out.wasSemicolon = true;
		    },
		    varcase(const StatType::Block<Out>&) {
			    out.newLine(); //Extra spacing
			    out.add('{').tabUpNewl();
			    genBlock(out, var);
			    out.unTabNewl().addNewl('}');
		    },
		    varcase(
		        const StatType::IfCond<Out>&) { genIfCond<false>(out, var); },
		    varcase(const StatType::While<Out>&) {
			    out.newLine(); //Extra spacing
			    out.add("while ");

			    genExprParens(out, var.cond);

			    out.newLine().add('{').tabUpNewl();

			    genBlock(out, var.bl);
			    out.unTabNewl().addNewl('}');
		    },
		    varcase(const StatType::RepeatUntil<Out>&) {
			    out.add("repeat");

			    out.newLine().add('{').tabUpNewl();

			    genBlock(out, var.bl);
			    out.unTabNewl();

			    out.add("} until");
			    genExpr(out, var.cond);
			    out.addNewl(';');

			    out.newLine(); //Extra spacing
			    out.wasSemicolon = true;
		    },
		    varcase(const StatType::ForIn<Out>&) {
			    out.add("for ");
			    genPat<true>(out, var.varNames);
			    out.add(" in ");
			    genExpr(out, var.exprs);
			    out.newLine().add('{').tabUpNewl();

			    genBlock(out, var.bl);
			    out.unTabNewl().addNewl('}');
		    },

		    varcase(const StatType::Fn&) { genFunc<false>(out, var, "fn "); },
		    varcase(const StatType::FnDecl<Out>&) {
			    genFunc<true>(out, var, "fn ");
		    },

		    varcase(const StatType::Function&) {
			    genFunc<false>(out, var, "function ");
		    },
		    varcase(const StatType::FunctionDecl<Out>&) {
			    genFunc<true>(out, var, "function ");
		    },

		    //Slu!

		    varcase(const StatType::Struct&) {
			    out.pushLocals(var.local2Mp);
			    genExSafety(out, var.exported, ast::OptSafety::DEFAULT);
			    out.add("struct ").add(out.db.asSv(var.name));
			    if (!var.params.empty())
			    {
				    out.add('(');
				    genParamList(out, var.params, false);
				    out.add(')');
			    }
			    out.add(' ');
			    genTable(out, var.type);
			    out.newLine();
			    out.popLocals();
		    },

		    varcase(const StatType::Union&) {
			    out.pushLocals(var.local2Mp);
			    genExSafety(out, var.exported, ast::OptSafety::DEFAULT);
			    out.add("union ").add(out.db.asSv(var.name));
			    if (!var.params.empty())
			    {
				    out.add('(');
				    genParamList(out, var.params, false);
				    out.add(')');
			    }
			    out.add(' ');
			    genTable(out, var.type);
			    out.newLine();
			    out.popLocals();
		    },

		    varcase(const StatType::ExternBlock<Out>&) {
			    out.newLine(); //Extra spacing
			    genSafety(out, var.safety);
			    out.add("extern ");
			    genString(out, var.abi);
			    out.add(" {").tabUpNewl();
			    for (auto& i : var.stats)
				    genStat(out, i);
			    out.unTabNewl().addNewl('}');
		    },

		    varcase(const StatType::UnsafeBlock<Out>&) {
			    out.newLine(); //Extra spacing
			    out.add("unsafe {").tabUpNewl();
			    for (auto& i : var.stats)
				    genStat(out, i);
			    out.unTabNewl().addNewl('}');
		    },
		    varcase(const StatType::UnsafeLabel) {
			    out.unTabTemp().add(":::unsafe:").tabUpTemp();
		    },
		    varcase(const StatType::SafeLabel) {
			    out.unTabTemp().add(":::safe:").tabUpTemp();
		    },

		    varcase(const StatType::Drop<Out>&) {
			    out.add("drop ");
			    ;
			    genExpr(out, var.expr);
			    out.addNewl(';');
			    out.wasSemicolon = true;
		    },
		    varcase(const StatType::Trait&) {
			    genExSafety(out, var.exported, ast::OptSafety::DEFAULT);
			    out.add("trait ").add(out.db.asSv(var.name)).add('(');
			    genParamList(out, var.params, false);
			    out.add(')');
			    if (var.whereSelf.has_value())
			    {
				    out.add(": ");
				    genTraitExpr(out, *var.whereSelf);
			    }
			    genWhereClauses(out, var.clauses);
			    out.add(" {").tabUpNewl().newLine();
			    for (const auto& i : var.itms)
				    genStat(out, i);
			    out.unTabNewl().addNewl('}');
		    },
		    varcase(const StatType::Impl&) {
			    genExSafety(out, var.exported,
			        var.isUnsafe ? ast::OptSafety::UNSAFE
			                     : ast::OptSafety::DEFAULT);
			    if (var.deferChecking)
				    out.add("defer ");
			    out.add("impl(");
			    genParamList(out, var.params, false);
			    out.add(") ");
			    if (var.forTrait.has_value())
			    {
				    genTraitExpr(out, *var.forTrait);
				    out.add(" for ");
			    }
			    genExpr(out, var.type);
			    genWhereClauses(out, var.clauses);
			    out.add(" {").tabUpNewl().newLine();
			    for (const auto& i : var.code)
				    genStat(out, i);
			    out.unTabNewl().addNewl('}');
		    },
		    varcase(const StatType::Use&) {
			    genExSafety(out, var.exported, ast::OptSafety::DEFAULT);
			    out.add("use ");
			    genModPath(out, out.db.asVmp(var.base));
			    genUseVariant(out, var.useVariant);
			    out.addNewl(';');
			    out.wasSemicolon = true;
		    },
		    varcase(const StatType::Mod<Out>&) {
			    genExSafety(out, var.exported, ast::OptSafety::DEFAULT);
			    out.add("mod ").add(out.db.asSv(var.name)).addNewl(';');
			    out.wasSemicolon = true;
		    },
		    varcase(const StatType::ModAs<Out>&) {
			    genExSafety(out, var.exported, ast::OptSafety::DEFAULT);
			    out.add("mod ").add(out.db.asSv(var.name)).add(" as {");
			    out.tabUpNewl().newLine();

			    for (auto& i : var.code)
				    genStat(out, i);
			    out.unTabNewl().addNewl('}');
		    });
	}


	export template<AnyOutput Out> void genFile(Out& out, const ParsedFile& obj)
	{
		for (const auto& i : obj.code)
			genStat(out, i);
	}
} //namespace slu::parse
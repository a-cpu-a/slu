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
		case BinOpType::MAKE_RESULT:
			return "~~"sv;
		case BinOpType::UNION:
			return "||"sv;
		case BinOpType::AS:
			return "as"sv;
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
	inline void genTable(Out& out, const Table<Out>& obj)
	{
		out.add('{')
			.template tabUpNewl<false>();

		for (const Field<Out>& f : obj)
		{
			ezmatch(f)(
			varcase(const FieldType::NONE) { _ASSERT(false); },

			varcase(const FieldType::Expr2Expr<Out>&) {
				out.addIndent();
				out.add('[');
				genExpr(out, var.idx);
				out.add("] = ");
				genExpr(out, var.v);
			},
			varcase(const FieldType::Name2Expr<Out>&) {
				out.addIndent();
				out.add(out.db.asSv(var.idx))
					.add(" = ");
				genExpr(out, var.v);
			},
			varcase(const FieldType::Expr<Out>&) {
				out.addIndent();
				genExpr(out, var);
			}
			);
			out.template addNewl<false>(',');
		}

		out.unTab()
			.add('}');
	}
	inline void writeU64Hex(AnyOutput auto& out, const uint64_t v) {
		for (size_t i = 0; i < 16; i++)
		{
			const uint8_t c = (uint64_t(v) >> (60 - 4 * i)) & 0xF;
			out.add(numToHex(c));
		}
	}

	template<AnyOutput Out>
	inline void genExprParens(Out& out, const Expr<Out>& obj)
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
	inline void genExSafety(Out& out,const bool exported, const OptSafety& safety)
	{
		if (exported)out.add("ex ");
		genSafety(out, safety);
	}

	template<AnyOutput Out>
	inline void genTraitExpr(Out& out, const TraitExpr& obj)
	{
		if constexpr(Out::settings() & sluSyn)
		{
			for (const parse::ExprV<true>& i : obj.traitCombo)
			{
				genExpr(out, i);
				if (&i != &obj.traitCombo.back())
					out.add(" + ");
			}
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
	inline void genExpr(Out& out, const Expr<Out>& obj)
	{
		genUnOps(out,obj.unOps);

		using namespace std::literals;
		ezmatch(obj.data)(
		varcase(const ExprType::Parens<Out>&) {
			out.add('(');
			genExpr(out, *var);
			out.add(')');
		},
		varcase(const ExprType::MpRoot) {
			out.add(":>"sv);
		},
		varcase(const ExprType::Global<Out>) {
			genModPath(out, out.db.asVmp(var));
		},
		varcase(const ExprType::Local) {
			genNameOrLocal<true>(out, var);
		},
		varcase(const ExprType::Deref) {
			if constexpr (Out::settings() & sluSyn)
			{
				genExpr(out, *var.v);
				out.add(".*");
			}
		},
		varcase(const ExprType::Index<Out>&) {
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
		varcase(const ExprType::Call<Out>&) {
			genCall<true>(out, var);
		},
		varcase(const ExprType::SelfCall<Out>&) {
			genSelfCall<true>(out, var);
		},
		varcase(const ExprType::Nil) {
			out.add("nil"sv);
		},
		varcase(const ExprType::False) {
			out.add("false"sv);
		},
		varcase(const ExprType::True) {
			out.add("true"sv);
		},
		varcase(const ExprType::VarArgs) {
			out.add("..."sv);
		},
		varcase(const ExprType::F64) {
			if (isinf(var) && var>0.0f)
				out.add("1e999");
			else
				out.add(std::to_string(var));
		},

		varcase(const ExprType::I64) {
			_ASSERT(!(Out::settings() & parse::noIntOverflow) || var >= 0);
			if (var < 0)
			{
				out.add("0x");
				writeU64Hex(out,var);
			}
			else
				out.add(std::to_string(var));
		},

		varcase(const ExprType::String&) {
			genString(out,var.v);
		},
		varcase(const ExprType::Function<Out>&) {
			out.add("function ");
			genFuncDef(out, var,""sv);
		},
		varcase(const ExprType::Call<Out>&) {
			genCall<true>(out, var);
		},
		varcase(const ExprType::SelfCall<Out>&) {
			genSelfCall<true>(out, var);
		},
		varcase(const ExprType::Table<Out>&) {
			genTable(out, var);
		},
		varcase(const ExprType::MultiOp<Out>&) {
			genExpr(out, *var.first);
			for (const auto& [op,ex] : var.extra)
			{
				out.add(' ')
					.add(getBinOpAsStr<Out>(op))
					.add(' ');
				genExpr(out, ex);
			}
		},
		varcase(const ExprType::OpenRange) {
			out.add("..");
		},
		varcase(const ExprType::Lifetime&) {
			if constexpr (Out::settings() & sluSyn)
				genLifetime(out, var);
		},
		varcase(const ExprType::TraitExpr&) {
			genTraitExpr(out, var);
		},
		varcase(const ExprType::IfCond<Out>&) {
			genIfCond<true>(out, var);
		},
		varcase(const ExprType::PatTypePrefix&) {},//Yes, nothing
		varcase(const ExprType::U64) {
			out.add(std::to_string(var));
		},
		varcase(const ExprType::P128) {
			out.add(parse::u128ToStr(var.lo, var.hi));
		},
		varcase(const ExprType::M128) {
			out.add(" -").add(parse::u128ToStr(var.lo, var.hi));
		},

		varcase(const ExprType::Inferr) {
			out.add('?');
		},
		varcase(const ExprType::Dyn&) {
			out.add("dyn ");
			genTraitExpr(out, var.expr);
		},
		varcase(const ExprType::Impl&) {
			out.add("impl ");
			genTraitExpr(out, var.expr);
		},
		varcase(const ExprType::Err&) {
			if constexpr (Out::settings() & sluSyn)
			{
				out.add("~~");
				genExpr(out, *var.err);
			}
		},
		varcase(const ExprType::Slice&) {
			if constexpr (Out::settings() & sluSyn)
			{
				out.add('[');
				genExpr(out, *var.v);
				out.add(']');
			}
		},
		varcase(const ExprType::Union&) {
			if constexpr (Out::settings() & sluSyn)
			{
				out.add("union ");
				genTable(out, var.fields);
			}
		},
		varcase(const ExprType::FnType&) {
			if constexpr (Out::settings() & sluSyn)
			{
				genSafety(out, var.safety);
				out.add("fn ");
				genExpr(out, *var.argType);
				out.add(" -> ");
				genExpr(out, *var.retType);
			}
		}
		);
		if constexpr(Out::settings()&sluSyn)
		{
			for (const PostUnOpType t : obj.postUnOps)
				out.add(getPostUnOpAsStr(t));
		}
	}
	template<AnyOutput Out>
	inline void genExprList(Out& out, const ExprList<Out>& obj)
	{
		for (const Expr<Out>& e : obj)
		{
			genExpr(out, e);
			if (&e != &obj.back())
				out.add(", ");
		}
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
	inline void genArgs(Out& out, const Args<Out>& itm)
	{
		//if (!arg.funcName.empty())
		//{
		//	out.add(':')
		//		.add(out.db.asSv(arg.funcName));
		//}
		ezmatch(itm)(
		varcase(const ArgsType::ExprList<Out>&) {
			out.add('(');
			genExprList(out, var);
			out.add(')');
		},
		varcase(const ArgsType::Table<Out>&) {
			genTable(out, var);
		},
		varcase(const ArgsType::String&) {
			out.add(' '); genString(out, var.v);
		}
		);
	}

	template<bool boxed,AnyOutput Out>
	inline void genCall(Out& out, const Call<Out,boxed>& itm)
	{
		genExpr(out, *itm.v);
		genArgs(out, itm.args);
	}
	template<bool boxed,AnyOutput Out>
	inline void genSelfCall(Out& out, const SelfCall<Out,boxed>& itm)
	{
		genExpr(out, *itm.v);
		out.add(sel<Out>(":","."))
			.add(out.db.asSv(itm.method));
		genArgs(out, itm.args);
	}

	/*template<AnyOutput Out>
	inline void genSubVar(Out& out, const SubVar<Out>& obj)
	{
		for (const ArgFuncCall<Out>& arg : obj.funcCalls)
		{
			genArgFuncCall(out, arg);
		}
		ezmatch(obj.idx)(
		varcase(const SubVarType::Expr<Out>&) {
			out.add('[');
			genExpr(out, var);
			out.add(']');
		},
		varcase(const SubVarType::NAME<Out>&) {
			const std::string_view txt = out.db.asSv(var.idx);
			if (txt.empty())return;
			out.add('.')
				.add(txt);
		},
		varcase(const SubVarType::Deref) {
			out.add(".*");
		}
		);
	}*/

	template<AnyOutput Out>
	inline void genModPath(Out& out, const lang::ViewModPath& obj)
	{
		out.add(":>::");
		out.add(obj[0]);
		for (size_t i = 1; i < obj.size(); i++)
		{
			out.add("::");
			out.add(obj[i]);
		}
	}
	//template<AnyOutput Out>
	//inline void genVar(Out& out, const Var<Out>& obj)
	//{
	//	ezmatch(obj.base)(
	//	varcase(const BaseVarType::Root) {
	//		out.add(":>");
	//	},
	//	varcase(const BaseVarType::Local) {
	//		//TODO
	//	},
	//	varcase(const BaseVarType::NAME<Out>&) {
	//		out.add(out.db.asSv(var.v));
	//	},
	//	varcase(const BaseVarType::Expr<Out>&) {
	//		out.add('(');
	//		genExpr(out, var);
	//		out.add(')');
	//	}
	//	);
	//	for (const SubVar<Out>& sub :  obj.sub)
	//	{
	//		genSubVar(out, sub);
	//	}
	//}
	template<AnyOutput Out>
	inline void genParamList(Out& out, const ParamList<Out>& itm,const bool hasVarArgParam)
	{
		for (const Parameter<Out>& par : itm)
		{
			genNameOrLocal<Out::settings()& sluSyn>(out, par.name);
			if constexpr (Out::settings() & sluSyn)
			{
				out.add(" = ");
				genExpr(out, par.type);
			}

			if (&par != &itm.back() || hasVarArgParam)
				out.add(", ");
		}
		if (hasVarArgParam)
			out.add("...");
	}
	template<bool isDecl,AnyOutput Out>
	inline void genFunc(Out& out, const auto& itm, const std::string_view kw)
	{
		if constexpr (isDecl)
		{
			if constexpr (Out::settings() & sluSyn)
			{
				genExSafety(out, itm.exported, itm.safety);
				out.pushLocals(itm.local2Mp);
			}

			out.add(kw);
			genFuncDecl(out, itm, out.db.asSv(itm.name));
			out.addNewl(";");
			out.wasSemicolon = true;
			if constexpr (Out::settings() & sluSyn)
				out.popLocals();
		}
		else
		{
			if constexpr (Out::settings() & sluSyn)
				genExSafety(out, itm.exported, itm.func.safety);

			out.add(kw);
			genFuncDef(out, itm.func, out.db.asSv(itm.name));
		}
	}
	template<AnyOutput Out>
	inline void genFuncDecl(Out& out, const auto& itm, const std::string_view name)
	{
		out.add(name);
		out.add('(');
		genParamList(out, itm.params, itm.hasVarArgParam);
		out.add(')');

		if constexpr (Out::settings() & sluSyn)
		{
			if (itm.retType.has_value())
			{
				out.add(" -> ");
				genExpr(out, **itm.retType);
			}
		}
	}
	template<AnyOutput Out>
	inline void genFuncDef(Out& out, const Function<Out>& var,const std::string_view name)
	{
		if constexpr (Out::settings() & sluSyn)
			out.pushLocals(var.local2Mp);

		genFuncDecl(out, var, name);

		if constexpr (out.settings() & sluSyn)
			out.newLine().add('{');
		out.tabUpNewl();

		genBlock(out, var.block);

		out.unTabNewl()
			.addNewl(sel<Out>("end","}"));

		out.newLine();//Extra spacing

		if constexpr (Out::settings() & sluSyn)
			out.popLocals();
	}

	template<AnyOutput Out>
	inline void genDestrSpec(Out& out, const DestrSpec<Out>& obj)
	{
		ezmatch(obj)(
		varcase(const DestrSpecType::Spat<Out>&) {
			genExpr(out, var);
			out.add(' ');
		},
		varcase(const DestrSpecType::Prefix&) {
			genUnOps(out, var);
		}
		);
	}
	template<bool isLocal,AnyOutput Out>
	inline void genNameOrLocal(Out& out, const LocalOrName<Out, isLocal>& obj)
	{
		if constexpr (isLocal)
			out.add(out.db.asSv(out.resolveLocal(obj)));
		else
			out.add(out.db.asSv(obj));
	}
	template<bool isLocal,AnyOutput Out>
	inline void genPat(Out& out, const Pat<Out, isLocal>& obj)
	{
		ezmatch(obj)(
		varcase(const PatType::DestrAny) {
			out.add('_');
		},
		varcase(const PatType::Simple<Out>&) {
			genExpr(out, var);
		},
			// Fields / List
		varcase(const auto&)requires AnyCompoundDestr<isLocal,std::remove_cvref_t<decltype(var)>> {
			genDestrSpec(out, var.spec);
			out.add('{').tabUpNewl();

			constexpr bool isList = std::is_same_v<decltype(var), const PatType::DestrList<Out,isLocal>&>;

			for (const auto& field : var.items)
			{
				if constexpr(isList)
					genPat<isLocal>(out, field);
				else
				{
					out.add('|')
						.add(out.db.asSv(field.name))
						.add("| ");

					genPat<isLocal>(out, field.pat);
				}
				if (&field != &var.items.back())
					out.add(',').newLine();
			}

			if(var.extraFields)
				out.add(", ..");
			out.unTabNewl().add('}');
			if(!var.name.empty())
			{
				out.add(' ');
				genNameOrLocal<isLocal>(out, var.name);
			}
		},
		varcase(const PatType::DestrName<Out,isLocal>&) {
			genDestrSpec(out, var.spec);
			genNameOrLocal<isLocal>(out, var.name);
		},
		varcase(const PatType::DestrNameRestrict<Out, isLocal>&) {
			genDestrSpec(out, var.spec);
			genNameOrLocal<isLocal>(out, var.name);
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
	inline void genSoe(Out& out, const parse::Soe<Out>& obj)

	{
		ezmatch(obj)(
		varcase(const parse::SoeType::Block<Out>&) {
			out.newLine().add('{').tabUpNewl();
			genBlock(out, var);
			out.unTabNewl().add('}');
		},
		varcase(const parse::SoeType::Expr<Out>&) {
			out.add(" => ");
			genExpr(out, *var);
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
			genSoe(out, *itm.bl);
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
					genSoe(out, bl);
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
				genSoe(out, **itm.elseBlock);
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

	template<bool isLocal,size_t N,AnyOutput Out>
	inline void genVarStat(Out& out, const auto& obj,const char(&kw)[N])
	{
		if constexpr (Out::settings() & sluSyn)
		{
			if (obj.exported)
				out.add("ex ");
		}

		out.add(kw);
		if constexpr (Out::settings() & sluSyn)
			genPat<isLocal>(out, obj.names);
		else
			genAtribNameList(out, obj.names);
		if (!obj.exprs.empty())
		{
			out.add(" = ");
			genExprList(out, obj.exprs);
		}
		out.addNewl(';');
		out.wasSemicolon = true;
	}
	template<AnyOutput Out>
	inline void genStat(Out& out, const Statement<Out>& obj)
	{
		ezmatch(obj.data)(

		varcase(const StatementType::Semicol) {
			if(!out.wasSemicolon)
				out.add(';');
			out.wasSemicolon = true;
		},

		varcase(const StatementType::Assign<Out>&) {
			genExprList(out, var.vars);
			out.add(" = ");
			genExprList(out, var.exprs);
			out.addNewl(';');
			out.wasSemicolon = true;
		},
		varcase(const StatementType::Local<Out>&) {
			genVarStat<true>(out, var,"local ");
		},
		varcase(const StatementType::Let<Out>&) {
			genVarStat<true>(out, var, "let ");
		},
		varcase(const StatementType::Const<Out>&) {
			out.pushLocals(var.local2Mp);
			genVarStat<false>(out, var, "const ");
			out.popLocals();
		},
		varcase(const StatementType::CanonicLocal&) {
			if constexpr (Out::settings() & sluSyn)
			{
				genExSafety(out, var.exported, OptSafety::DEFAULT);
				out.add("let ");
				//genExpr(out, var.type); //TODO: gen resolved types?
				//out.add(' ');
				genNameOrLocal<true>(out, var.name);
				out.add(" = ");
				genExpr(out, var.value);
				out.addNewl(';');
				out.wasSemicolon = true;
			}
		},
		varcase(const StatementType::CanonicGlobal&) {
			if constexpr (Out::settings() & sluSyn)
			{
				out.pushLocals(var.local2Mp);
				genExSafety(out, var.exported, OptSafety::DEFAULT);
				out.add("const ");
				//genExpr(out, var.type); //TODO: gen resolved types?
				//out.add(' ');
				genNameOrLocal<false>(out, var.name);
				out.add(" = ");
				genExpr(out, var.value);
				out.addNewl(';');
				out.wasSemicolon = true;
				out.popLocals();
			}
		},

		varcase(const StatementType::Call<Out>&) {
			genCall<false>(out, var);
			out.addNewl(';');
			out.wasSemicolon = true;
		},
		varcase(const StatementType::SelfCall<Out>&) {
			genSelfCall<false>(out, var);
			out.addNewl(';');
			out.wasSemicolon = true;
		},

		varcase(const StatementType::Label<Out>&) {
			out.unTabTemp()
				.add(sel<Out>("::", ":::"))
				.add(out.db.asSv(var.v))
				.addNewl(sel<Out>("::", ":"))
				.tabUpTemp();
		},
		varcase(const StatementType::Break) {
			out.addNewl("break;");
			out.wasSemicolon = true;
		},
		varcase(const StatementType::Goto<Out>&) {
			out.add("goto ")
				.add(out.db.asSv(var.v))
				.addNewl(';');
			out.wasSemicolon = true;
		},
		varcase(const StatementType::Block<Out>&) {
			out.newLine();//Extra spacing
			out.add(sel<Out>("do","{"))
				.tabUpNewl();
			genBlock(out, var);
			out.unTabNewl()
				.addNewl(sel<Out>("end", "}"));
		},
		varcase(const StatementType::IfCond<Out>&) {
			genIfCond<false>(out, var);
		},
		varcase(const StatementType::While<Out>&) {
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
		varcase(const StatementType::RepeatUntil<Out>&) {
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
		varcase(const StatementType::ForNum<Out>&) {

			out.add("for ");
			if constexpr (Out::settings() & sluSyn)
				genPat<true>(out, var.varName);
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
		varcase(const StatementType::ForIn<Out>&) {
			out.add("for ");
			if constexpr (Out::settings() & sluSyn)
				genPat<true>(out, var.varNames);
			else
				genNames(out, var.varNames);
			out.add(" in ");
			if constexpr (Out::settings() & sluSyn)
				genExpr(out, var.exprs);
			else
				genExprList(out, var.exprs);

			if constexpr (Out::settings() & sluSyn)
				out.newLine().add('{');
			else
				out.add(" do");
			out.tabUpNewl();

			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl(sel<Out>("end", "}"));
		},

		varcase(const StatementType::Fn<Out>&) {
			genFunc<false>(out, var, "fn ");
		},
		varcase(const StatementType::FnDecl<Out>&) {
			genFunc<true>(out, var, "fn ");
		},

		varcase(const StatementType::Function<Out>&) {
			genFunc<false>(out, var, "function ");
		},
		varcase(const StatementType::FunctionDecl<Out>&) {
			genFunc<true>(out, var, "function ");
		},
		varcase(const StatementType::LocalFunctionDef<Out>&) {
			out.add("local function ");
			genFuncDef(out, var.func, out.db.asSv(var.name));
		},

		//Slu!

		varcase(const StatementType::Struct&) {
			if constexpr (Out::settings() & sluSyn)
			{
				out.pushLocals(var.local2Mp);
				if (var.exported)out.add("ex ");
				out.add("struct ").add(out.db.asSv(var.name));
				if (!var.params.empty())
				{
					out.add('(');
					genParamList(out, var.params, false);
					out.add(')');
				}
				if (!var.type.isBasicStruct())
					out.add(" = ");
				else
					out.add(' ');

				genExpr(out, var.type);
				out.newLine();
				out.popLocals();
			}
		},

		varcase(const StatementType::Union&) {
			if constexpr (Out::settings() & sluSyn)
			{
				out.pushLocals(var.local2Mp);
				if (var.exported)out.add("ex ");
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
			}
		},

		varcase(const StatementType::ExternBlock<Out>&) {
			out.newLine();//Extra spacing
			genSafety(out, var.safety);
			out.add("extern ");
			genString(out, var.abi);
			out.add(" {").tabUpNewl();
			for (auto& i : var.stats)
				genStat(out, i);
			out.unTabNewl()
				.addNewl('}');
		},

		varcase(const StatementType::UnsafeBlock<Out>&) {
			out.newLine();//Extra spacing
			out.add("unsafe {")
				.tabUpNewl();
			genBlock(out, var.bl);
			out.unTabNewl()
				.addNewl('}');
		},
		varcase(const StatementType::UnsafeLabel) {
			out.unTabTemp()
				.add(":::unsafe:")
				.tabUpTemp();
		},
		varcase(const StatementType::SafeLabel) {
			out.unTabTemp()
				.add(":::safe:")
				.tabUpTemp();
		},

		varcase(const StatementType::Drop<Out>&) {
			out.add("drop ");;
			genExpr(out, var.expr); 
			out.addNewl(';');
			out.wasSemicolon = true;
		},
		varcase(const StatementType::Use&) {
			if constexpr (Out::settings() & sluSyn)
			{
				if (var.exported)out.add("ex ");
				out.add("use ");
				genModPath(out, out.db.asVmp(var.base));
				genUseVariant(out, var.useVariant);
				out.addNewl(';');
				out.wasSemicolon = true;
			}
		},
		varcase(const StatementType::Mod<Out>&) {
			if (var.exported)out.add("ex ");
			out.add("mod ").add(out.db.asSv(var.name)).addNewl(';');
			out.wasSemicolon = true;
		},
		varcase(const StatementType::ModAs<Out>&) {
			if (var.exported)out.add("ex ");
			out.add("mod ").add(out.db.asSv(var.name)).add(" as {");
			out.tabUpNewl().newLine();

			genBlock(out,var.bl);
			out.unTabNewl().add('}');
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
				out.add(' ');
				genExprList(out, obj.retExprs);
			}
			out.add(';');
			out.wasSemicolon = true;
		}
	}

	template<AnyOutput Out>
	inline void genFile(Out& out,const ParsedFile<Out>& obj)
	{
		if constexpr (Out::settings() & sluSyn)
		{
			for (const auto& i : obj.code)
				genStat(out, i);
		}
		else
		{
			out.pushLocals(obj.local2Mp);
			genBlock(out, obj.code);
			out.popLocals();
		}
	}
}
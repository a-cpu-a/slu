module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <slu/ext/CppMatch.hpp>
#include <slu/Panic.hpp>
export module slu.paint.paint;

import slu.char_info;
import slu.settings;
import slu.ast.enums;
import slu.ast.state;
import slu.ast.state_decls;
import slu.paint.basics;
import slu.paint.paint_ops;
import slu.paint.sem_output;
import slu.parse.input;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::paint
{
	template<bool SKIP_SPACE = true>
	inline void paintString(AnySemOutput auto& se, const std::string_view sv,
	    const ast::Position end, const Tok tint)
	{
		if constexpr (SKIP_SPACE)
			skipSpace(se);

		se.template add<Tok::STRING_OUT>(tint);
		const char fChar = se.in.get();
		if (fChar == '[')
		{
			size_t level = 0;
			while (se.in.peek() == '=')
			{
				level++;
				se.template add<Tok::STRING_OUT>(tint);
				se.in.skip();
			}
			se.template add<Tok::STRING_OUT>(tint);
			se.in.skip();

			se.template move<Tok::STRING>(end);
			se.template replPrev<Tok::STRING_OUT>(tint, level + 2);
		} else
		{
			se.template move<Tok::STRING>(end);
			se.template replPrev<Tok::STRING_OUT>(tint);
		}
	}
	template<bool SKIP_SPACE = true>
	inline void paintNumber(AnySemOutput auto& se, const Tok tint)
	{
		if constexpr (SKIP_SPACE)
			skipSpace(se);

		// null, if right before the end of the file.
		const char ch = se.in.isOob(1) ? 0 : se.in.peekAt(1);
		bool hex = ch == 'x' || ch == 'X';
		if (hex || ch == 'O' || ch == 'o' || ch == 'd' || ch == 'D')
		{
			se.template add<Tok::NUMBER_KIND>(tint, 2);
			se.in.skip(2);
		}
		bool wasUscore = false;
		bool parseType = false;
		while (se.in)
		{
			const char chr = se.in.peek();
			if (wasUscore && chr != '_' && (!hex || !slu::isHexDigitChar(chr))
			    && slu::isValidNameStartChar(chr))
			{
				parseType = true;
				break;
			}
			wasUscore = false;
			if (!hex && (chr == 'e' || chr == 'E')
			    || hex && (chr == 'p' || chr == 'P'))
			{
				se.template add<Tok::NUMBER_KIND>(tint, 1);
				se.in.skip();
				const char maybeSign = se.in.peek();
				if (maybeSign == '+' || maybeSign == '-')
				{
					se.template add<Tok::NUMBER_KIND>(tint, 1);
					se.in.skip();
				}
				continue;
			}
			if (chr == '_')
				wasUscore = true;

			if (chr == '.')
			{
				// Handle ranges, concat, etc
				if (se.in.peekAt(1) == '.')
					break;
			}

			if (chr != '.' && chr != '_' && !(hex && slu::isHexDigitChar(chr))
			    && !slu::isDigitChar(chr))
				break;

			se.template add<Tok::NUMBER>(tint);
			se.in.skip();
		}
		if (parseType)
		{
			while (se.in)
			{
				if (slu::isValidNameChar(se.in.peek()))
				{
					se.template add<Tok::NUMBER_TYPE>(tint);
					se.in.skip();
				} else
					break;
			}
		}
	}
	template<Tok nameTok, AnySemOutput Se>
	inline void paintField(Se& se, const parse::Field<Se>& itm)
	{
		ezmatch(itm)(
		    varcase(const parse::FieldType::Expr2Expr&) {
			    paintKw<Tok::PUNCTUATION>(se, "[");
			    paintExpr(se, var.idx);
			    paintKw<Tok::PUNCTUATION>(se, "]");
			    paintKw<Tok::ASSIGN>(se, "=");
			    paintExpr<nameTok>(se, var.v);
		    },
		    varcase(const parse::FieldType::Name2Expr&) {
			    paintName<Tok::NAME_TABLE>(se, var.idx);
			    paintKw<Tok::ASSIGN>(se, "=");
			    paintExpr<nameTok>(se, var.v);
		    },
		    varcase(
		        const parse::FieldType::Expr&) { paintExpr<nameTok>(se, var); },
		    varcase(const parse::FieldType::NONE) {
			    Slu_panic(
			        "field shouldnt be FieldType::NONE, found while painting.");
		    });
	}
	template<Tok nameTok, AnySemOutput Se>
	inline void paintTable(Se& se, const parse::Table<Se>& itm)
	{
		paintKw<Tok::GEN_OP>(se, "{");
		for (const parse::Field<Se>& f : itm)
		{
			paintField<nameTok>(se, f);
			skipSpace(se);
			if (se.in.peek() == ',')
				paintKw<Tok::PUNCTUATION>(se, ",");
			else if (se.in.peek() == ';')
				paintKw<Tok::PUNCTUATION>(se, ";");
		}
		paintKw<Tok::GEN_OP>(se, "}");
	}
	template<AnySemOutput Se>
	inline void paintTypeExpr(Se& se, const parse::Expr& itm,
	    const Tok tint = Tok::NONE, const bool unOps = true)
	{
		paintExpr<Tok::NAME_TYPE>(se, itm, tint, unOps);
	}
	template<Tok nameTok = Tok::NAME, AnySemOutput Se>
	inline void paintExpr(Se& se, const parse::Expr& itm,
	    const Tok tint = Tok::NONE, const bool unOps = true)
	{
		se.move(itm.place);
		/*if (std::holds_alternative<parse::ExprType::MultiOp>(itm.data))
		{
		    //complex version
		}*/
		if (unOps)
		{
			for (const auto& i : itm.unOps)
				paintUnOpItem(se, i);
		}
		paintExprData<nameTok>(se, itm.data, tint);
		if (unOps)
		{
			for (const auto& i : itm.postUnOps)
				paintPostUnOp(se, i);
		}
	}
	template<Tok nameTok = Tok::NAME, AnySemOutput Se>
	inline void paintExprData(
	    Se& se, const parse::ExprData<Se>& itm, const Tok tint = Tok::NONE)
	{
		ezmatch(itm)(
		    varcase(const parse::ExprType::MpRoot) {
			    paintKw<Tok::MP_ROOT>(se, ":>");
		    },
		    varcase(const parse::ExprType::Local) {
			    paintNameOrLocal<true, nameTok>(se, var);
		    },
		    varcase(const parse::ExprType::Global<Se>&) {
			    paintMp<nameTok>(se, var);
		    },
		    varcase(const parse::ExprType::Parens<Se>&) {
			    paintKw<Tok::GEN_OP>(se, "(");
			    paintExpr(se, *var);
			    paintKw<Tok::GEN_OP>(se, ")");
		    },
		    varcase(const parse::ExprType::Deref&) {
			    paintExpr(se, *var.v);
			    paintKw<Tok::DEREF>(se, ".*");
		    },
		    varcase(const parse::ExprType::Index&) {
			    paintExpr(se, *var.v);
			    paintKw<Tok::GEN_OP>(se, "[");
			    paintExpr(se, *var.idx);
			    paintKw<Tok::GEN_OP>(se, "]");
		    },
		    varcase(const parse::ExprType::Field<Se>&) {
			    paintExpr(se, *var.v);
			    paintKw<Tok::GEN_OP>(se, ".");
			    paintPoolStr(se, var.field);
		    },
		    varcase(const parse::ExprType::SelfCall&) {
			    paintSelfCall<true>(se, var);
		    },
		    varcase(const parse::ExprType::Call&) { paintCall<true>(se, var); },
		    varcase(const parse::ExprType::MultiOp<Se>&) {
			    paintExpr<nameTok>(se, *var.first);
			    for (const auto& [op, expr] : var.extra)
			    {
				    paintBinOp(se, op);
				    paintExpr<nameTok>(se, expr);
			    }
		    },
		    varcase(const parse::ExprType::Nil) {
			    paintKw<Tok::BUILITIN_VAR>(se, "nil");
		    },
		    varcase(const parse::ExprType::VarArgs) {
			    paintKw<Tok::PUNCTUATION>(se, "...");
		    },
		    varcase(const parse::ExprType::OpenRange) {
			    paintKw<Tok::RANGE>(se, "..");
		    },
		    varcase(const parse::ExprType::String&) {
			    paintString(se, var.v, var.end, tint);
		    },
		    varcase(const parse::ExprType::Unfinished&) {
			    paintKw<Tok::COMMENT_WIP>(se,
			        "TO"
			        "DO!");
			    paintString(se, var.msg, var.end, tint);
		    },
		    varcase(const parse::ExprType::CompBuiltin&) {
			    paintKw<Tok::COMMENT_WIP>(se,
			        "_COMP_TO"
			        "DO!");
			    paintKw<Tok::COMMENT_WIP>(se, "(");
			    paintString(se, var.kind, var.kindEnd, slu::paint::Tok::NONE);
			    paintKw<Tok::PUNCTUATION>(se, ",");
			    paintExpr(se, *var.value);
			    paintKw<Tok::COMMENT_WIP>(se, ")");
		    },
		    varcase(const parse::ExprType::F64) { paintNumber(se, tint); },
		    varcase(const parse::ExprType::P128) { paintNumber(se, tint); },
		    varcase(const parse::ExprType::M128) { paintNumber(se, tint); },
		    varcase(const parse::ExprType::I64) { paintNumber(se, tint); },
		    varcase(const parse::ExprType::U64) { paintNumber(se, tint); },
		    varcase(const parse::ExprType::IfCond<Se>&) {
			    paintIfCond<true>(se, var);
		    },
		    varcase(
		        const parse::ExprType::Lifetime&) { paintLifetime(se, var); },
		    varcase(
		        const parse::ExprType::TraitExpr&) { paintTraitExpr(se, var); },
		    varcase(const parse::ExprType::Table<Se>&) {
			    paintTable<nameTok>(se, var);
		    },
		    varcase(const parse::ExprType::Function&) {
			    paintFuncDef(se, var, lang::MpItmId::newEmpty(), false);
		    },
		    varcase(const parse::ExprType::PatTypePrefix&) {
			    Slu_panic("Pat type prefix leaked outside of pattern parsing!");
		    },

		    varcase(const parse::ExprType::Infer) {
			    paintKw<Tok::GEN_OP>(se, "?");
		    },
		    varcase(const parse::ExprType::Err&) {
			    paintKw<Tok::GEN_OP>(se, "~~");
			    paintTypeExpr(se, *var.err, tint);
		    },
		    varcase(const parse::ExprType::Struct&) {
			    paintKw<Tok::CON_STAT>(se, "struct");
			    paintTable<Tok::NAME_TYPE>(se, var.fields);
		    },
		    varcase(const parse::ExprType::Union&) {
			    paintKw<Tok::CON_STAT>(se, "union");
			    paintTable<Tok::NAME_TYPE>(se, var.fields);
		    },
		    varcase(const parse::ExprType::Dyn&) {
			    paintKw<Tok::DYN>(se, "dyn");
			    paintTraitExpr(se, var.expr);
		    },
		    varcase(const parse::ExprType::Impl&) {
			    paintKw<Tok::IMPL>(se, "impl");
			    paintTraitExpr(se, var.expr);
		    },
		    varcase(const parse::ExprType::FnType&) {
			    paintSafety(se, var.safety);
			    paintKw<Tok::FN_STAT>(se, "fn");
			    paintTypeExpr(se, *var.argType);
			    paintKw<Tok::GEN_OP>(se, "->");
			    paintTypeExpr(se, *var.retType);
		    });
	}
	template<Tok tok, Tok overlayTok = Tok::NONE, AnySemOutput Se>
	inline void paintMp(Se& se, const lang::MpItmId& itm)
	{
		skipSpace(se);
		if (parse::checkToken(se.in, ":>"))
		{
			paintKw<Tok::MP_ROOT, overlayTok>(se, ":>");
			paintKw<Tok::MP, overlayTok>(se, "::");
		}
		const lang::ViewModPath mp = se.in.genData.asVmp(itm);
		for (size_t i = 0; i < mp.size();)
		{
			if (parse::checkTextToken(se.in, "self"))
				paintKw<Tok::VAR_STAT, overlayTok>(se, "self");
			else if (parse::checkTextToken(se.in, "Self"))
				paintKw<Tok::CON_STAT, overlayTok>(se, "Self");
			else if (parse::checkTextToken(se.in, "crate"))
				paintKw<Tok::CON_STAT, overlayTok>(se, "crate");
			else
			{
				if (parse::checkTextToken(se.in, mp[i]))
					paintSv<tok, overlayTok>(se, mp[i]);
				else
				{
					i++;
					continue;
				}
				i++;
			}
			if (i >= mp.size())
				break;

			if (parse::checkToken(se.in, "::"))
				paintKw<Tok::MP, overlayTok>(se, "::");
			else
				break;
		}
	}
	template<bool isLocal, Tok nameTok, AnySemOutput Se>
	inline void paintDestrField(
	    Se& se, const parse::DestrField<Se, isLocal>& itm)
	{
		paintKw<Tok::GEN_OP>(se, "|");
		paintPoolStr<Tok::NAME_TABLE>(se, itm.name);
		paintKw<Tok::GEN_OP>(se, "|");
		paintPat<isLocal, nameTok>(se, itm.pat);
	}
	template<AnySemOutput Se>
	inline void paintTypePrefix(Se& se, const parse::TypePrefix& itm)
	{
		for (const parse::UnOpItem& i : itm)
			paintUnOpItem(se, i);
	}
	template<Tok nameTok, AnySemOutput Se>
	inline void paintDestrSpec(Se& se, const parse::DestrSpec& itm)
	{
		ezmatch(itm)(
		    varcase(const parse::DestrSpecType::Spat&) {
			    paintExpr<nameTok>(se, var);
		    },
		    varcase(const parse::DestrSpecType::Prefix&) {
			    paintTypePrefix(se, var);
		    });
	}
	template<bool isLocal, Tok nameTok, AnySemOutput Se>
	inline void paintPat(Se& se, const parse::Pat<Se, isLocal>& itm)
	{
		ezmatch(itm)(

		    //parse::PatType::DestrFields or parse::PatType::DestrList
		    varcase(const auto&) //requires(parse::AnyCompoundDestr<isLocal,
		                         //decltype(var)>)
		    {
			    paintDestrSpec<nameTok>(se, var.spec);
			    paintKw<Tok::GEN_OP>(se, "{");
			    for (const auto& i : var.items)
			    {
				    if constexpr (std::same_as<std::remove_cvref_t<decltype(i)>,
				                      parse::DestrFieldV<true, isLocal>>)
					    paintDestrField<isLocal, nameTok>(se, i);
				    else
					    paintPat<isLocal, nameTok>(se, i);

				    if (&i != &var.items.back())
					    paintKw<Tok::PUNCTUATION>(se, ",");
			    }
			    if (var.extraFields)
			    {
				    paintKw<Tok::PUNCTUATION>(se, ",");
				    paintKw<Tok::PUNCTUATION>(se, "..");
			    }
			    paintKw<Tok::GEN_OP>(se, "}");

			    paintNameOrLocal<isLocal, Tok::NAME>(se, var.name);
		    },
		    varcase(const parse::PatType::Simple<Se>&) { paintExpr(se, var); },
		    varcase(const parse::PatType::DestrAny<Se, isLocal>) {
			    paintKw<Tok::GEN_OP>(se, "_");
		    },
		    varcase(const parse::PatType::DestrName<Se, isLocal>&) {
			    paintDestrSpec<nameTok>(se, var.spec);
			    paintNameOrLocal<isLocal, Tok::NAME>(se, var.name);
		    },
		    varcase(const parse::PatType::DestrNameRestrict<Se, isLocal>&) {
			    paintDestrSpec<nameTok>(se, var.spec);
			    paintNameOrLocal<isLocal, Tok::NAME>(se, var.name);
			    paintKw<Tok::PAT_RESTRICT>(se, "=");
			    paintExpr(se, var.restriction);
		    });
	}
	template<Tok tok, AnySemOutput Se>
	inline void paintNameList(Se& se, const std::vector<lang::MpItmId>& itm)
	{
		for (const lang::MpItmId& i : itm)
		{
			paintName<tok>(se, i);

			if (&i != &itm.back())
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
	}
	template<Tok tok, AnySemOutput Se>
	inline void paintAttribNameList(
	    Se& se, const parse::AttribNameList<Se>& itm)
	{
		for (const parse::AttribName<Se>& i : itm)
		{
			paintName<tok>(se, i.name);
			if (!i.attrib.empty())
			{
				paintKw<Tok::PUNCTUATION>(se, "<");
				paintSv<tok>(se, i.attrib);
				paintKw<Tok::PUNCTUATION>(se, ">");
			}

			if (&i != &itm.back())
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
	}
	template<AnySemOutput Se>
	inline void paintArgs(Se& se, const parse::Args& itm)
	{
		ezmatch(itm)(
		    varcase(const parse::ArgsType::ExprList&) {
			    paintKw<Tok::GEN_OP>(se, "(");
			    paintExprList(se, var);
			    paintKw<Tok::GEN_OP>(se, ")");
		    },
		    varcase(const parse::ArgsType::Table<Se>&) {
			    paintTable<Tok::NAME>(se, var);
		    },
		    varcase(const parse::ArgsType::String&) {
			    paintString(se, var.v, var.end, Tok::NONE);
		    });
	}
	template<bool forCond, AnySemOutput Se>
	inline void paintEndBlock(
	    Se& se, const parse::Block<Se>& itm, const bool scopeOwner = true)
	{
		paintBlock(se, itm, scopeOwner);
		skipSpace(se);
		paintKw<Tok::BRACES>(se, "}");
	}
	template<AnySemOutput Se>
	inline void paintDoEndBlock(Se& se, const parse::Block<Se>& itm)
	{
		paintKw<Tok::BRACES>(se, "{");
		paintEndBlock<true>(se, itm);
	}
	template<AnySemOutput Se>
	inline void paintTraitExpr(Se& se, const parse::TraitExpr& itm)
	{
		skipSpace(se);
		se.move(itm.place);
		for (const parse::Expr& i : itm.traitCombo)
		{
			paintExpr<Tok::NAME_TRAIT>(se, i);
			if (&i != &itm.traitCombo.back())
				paintKw<Tok::ADD>(se, "+");
		}
	}
	template<AnySemOutput Se>
	inline void paintSafety(Se& se, const ast::OptSafety itm)
	{
		switch (itm)
		{
		case ast::OptSafety::SAFE:    paintKw<Tok::FN_STAT>(se, "safe"); break;
		case ast::OptSafety::UNSAFE:  paintKw<Tok::FN_STAT>(se, "unsafe"); break;
		case ast::OptSafety::DEFAULT:
		default:                      break;
		}
	}
	template<AnySemOutput Se>
	inline void paintStatOrRet(Se& se, const parse::Block<Se>& itm)
	{
		skipSpace(se);
		bool hadBrace = false;
		if (se.in.peek() == '{')
		{
			hadBrace = true;
			paintKw<Tok::BRACES>(se, "{");
		}

		paintBlock(se, itm, false);

		if (hadBrace)
			paintKw<Tok::BRACES>(se, "}");
		return;
	}
	template<AnySemOutput Se>
	inline void paintSoeOrBlock(Se& se, const parse::SoeOrBlock<Se>& itm)
	{
		ezmatch(itm)(
		    varcase(
		        const parse::SoeType::Block<Se>&) { paintStatOrRet(se, var); },
		    varcase(const parse::SoeType::Expr&) {
			    skipSpace(se);
			    if (se.in.peek() == '=')
				    paintKw<Tok::GEN_OP>(se, "=>");
			    paintExpr(se, *var);
		    });
	}
	template<AnySemOutput Se>
	inline void paintParamList(
	    Se& se, const parse::ParamList& itm, const bool hasVarArgParam)
	{
		for (const parse::Parameter& i : itm)
		{
			ezmatch(i.name)(
			    varcase(
			        const parse::LocalId&) { paintNameOrLocal<true>(se, var); },
			    varcase(const lang::MpItmId&) {
				    paintKw<Tok::VAR_STAT>(se, "const");
				    paintNameOrLocal<false>(se, var);
			    });
			paintKw<Tok::PAT_RESTRICT>(se, "=");
			paintTypeExpr(se, i.type);

			if (&i != &itm.back() || hasVarArgParam)
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
		if (hasVarArgParam)
			paintKw<Tok::PUNCTUATION>(se, "...");
	}
	template<Tok baseCol, AnySemOutput Se>
	inline void paintExportData(Se& se, lang::ExportData exported)
	{
		if (exported)
			paintKw<baseCol, Tok::EX_TINT>(se, "ex");
	}
	//Pos must be valid, unless the name is empty
	template<AnySemOutput Se>
	inline void paintFuncDecl(Se& se, const parse::SelfArg& selfArg,
	    const parse::ParamList& params, const bool hasVarArgParam,
	    const std::optional<std::unique_ptr<parse::Expr>>& retType,
	    const lang::MpItmId name, const lang::ExportData exported,
	    const ast::OptSafety safety, const ast::Position pos = {},
	    const bool fnKw = false)
	{
		paintExportData<Tok::FN_STAT>(se, exported);
		paintSafety(se, safety);
		if (fnKw)
			paintKw<Tok::FN_STAT>(se, "fn");
		else
			paintKw<Tok::FN_STAT>(se, "function");


		if (!name.empty())
		{
			paintName<Tok::NAME>(se, name);
			se.move(pos);
		}

		paintKw<Tok::GEN_OP>(se, "(");
		if (!selfArg.empty())
		{
			for (const ast::UnOpType t : selfArg.specifiers)
				paintUnOpItem(se, {{}, t});
			paintKw<Tok::VAR_STAT>(se, "self");
			if (!params.empty())
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
		paintParamList(se, params, hasVarArgParam);
		paintKw<Tok::GEN_OP>(se, ")");

		if (retType.has_value())
		{
			paintKw<Tok::GEN_OP>(se, "->");
			paintTypeExpr(se, **retType);
		}
	}
	template<bool isDecl, AnySemOutput Se>
	inline void paintFunc(Se& se, const auto& itm, const bool fnKw)
	{
		if constexpr (isDecl)
		{
			se.pushLocals(itm.local2Mp);
			paintFuncDecl(se, itm.selfArg, itm.params, itm.hasVarArgParam,
			    itm.retType, itm.name, itm.exported, itm.safety, itm.place,
			    fnKw);
			se.popLocals();
		} else
		{
			paintFuncDef(se, itm.func, itm.name, itm.exported, itm.place, fnKw);
		}
	}
	//Pos must be valid, unless the name is empty
	template<AnySemOutput Se>
	inline void paintFuncDef(Se& se, const parse::Function& func,
	    const lang::MpItmId name, const lang::ExportData exported,
	    const ast::Position pos = {}, const bool fnKw = false)
	{
		std::optional<std::unique_ptr<parse::Expr>> emptyTy{};
		const std::optional<std::unique_ptr<parse::Expr>>* retType;
		ast::OptSafety safety;
		retType = &func.retType;
		safety = func.safety;
		se.pushLocals(func.local2Mp);

		paintFuncDecl(se, func.selfArg, func.params, func.hasVarArgParam,
		    *retType, name, exported, safety, pos, fnKw);
		paintKw<Tok::BRACES>(se, "{");

		//No do, for functions in lua
		paintEndBlock<false>(se, func.block, false);

		se.popLocals();
	}
	template<bool isExpr, AnySemOutput Se>
	inline void paintIfCond(Se& se, const parse::BaseIfCond<Se, isExpr>& itm)
	{
		paintKw<Tok::COND_STAT>(se, "if");
		paintExpr(se, *itm.cond);

		paintSoeOrBlock(se, *itm.bl);
		for (const auto& [cond, bl] : itm.elseIfs)
		{
			paintKw<Tok::COND_STAT>(se, "else");
			paintKw<Tok::COND_STAT>(se, "if");
			paintExpr(se, cond);
			paintSoeOrBlock(se, bl);
		}
		if (itm.elseBlock.has_value())
		{
			paintKw<Tok::COND_STAT>(se, "else");
			paintSoeOrBlock(se, **itm.elseBlock);
		}
	}
	template<AnySemOutput Se>
	inline void paintUseVariant(Se& se, const parse::UseVariant& itm)
	{
		ezmatch(itm)(
		    // use xxxx::Xxx;
		    varcase(const parse::UseVariantType::IMPORT&){},

		    // use xxxx::Xxx::*;
		    varcase(const parse::UseVariantType::EVERYTHING_INSIDE) {
			    paintKw<Tok::MP>(se, "::");
			    paintKw<Tok::PUNCTUATION>(se, "*");
		    },
		    // use xxxx::Xxx as yyy;
		    varcase(const parse::UseVariantType::AS_NAME&) {
			    paintKw<Tok::VAR_STAT>(se, "as");
			    paintName<Tok::NAME>(se, var);
		    },
		    // use xxxx::Xxx::{self,a,b,c};
		    varcase(const parse::UseVariantType::LIST_OF_STUFF&) {
			    paintKw<Tok::MP>(se, "::");
			    paintKw<Tok::BRACES>(se, "{");
			    for (const lang::MpItmId& i : var)
			    {
				    paintMp<Tok::NAME>(se, i);

				    if (&i != &var.back())
					    paintKw<Tok::PUNCTUATION>(se, ",");
			    }
			    paintKw<Tok::BRACES>(se, "}");
		    });
	}
	template<bool isLocal, size_t TOK_SIZE, AnySemOutput Se>
	inline void paintVarStat(
	    Se& se, const auto& itm, const char (&tokChr)[TOK_SIZE])
	{
		paintExportData<Tok::VAR_STAT>(se, itm.exported);

		paintKw<Tok::VAR_STAT>(se, tokChr);

		paintPat<isLocal, Tok::NAME_TYPE>(se, itm.names);

		if (itm.exprs.empty())
			return;

		paintKw<Tok::ASSIGN>(se, "=");
		paintExprList(se, itm.exprs);
	}
	template<size_t TOK_SIZE, AnySemOutput Se>
	inline void paintStructBasic(
	    Se& se, const auto& itm, const char (&tokChr)[TOK_SIZE])
	{
		paintExportData<Tok::CON_STAT>(se, itm.exported);

		paintKw<Tok::CON_STAT>(se, tokChr);

		paintName<Tok::NAME>(se, itm.name);

		skipSpace(se);
		if (se.in.peek() == '(')
		{
			paintKw<Tok::PUNCTUATION>(se, "(");
			paintParamList(se, itm.params, false);
			paintKw<Tok::PUNCTUATION>(se, ")");
		}
	}
	template<bool boxed, AnySemOutput Se>
	inline void paintCall(Se& se, const parse::Call<boxed>& itm)
	{
		paintExpr(se, *itm.v);
		paintArgs(se, itm.args);
	}
	template<bool boxed, AnySemOutput Se>
	inline void paintSelfCall(Se& se, const parse::SelfCall<boxed>& itm)
	{
		paintExpr(se, *itm.v);
		paintKw<Tok::GEN_OP>(se, ".");
		paintName(se, itm.method);
		paintArgs(se, itm.args);
	}

	template<AnySemOutput Se>
	inline void paintWhereClauses(Se& se, const parse::WhereClauses& itm)
	{
		if (itm.empty())
			return;
		paintKw<Tok::IMPL>(se, "where");

		for (const parse::WhereClause& i : itm)
		{
			paintName<Tok::NAME>(se, i.var);

			paintKw<Tok::GEN_OP>(se, ":");
			paintTraitExpr(se, i.bound);

			if (&i != &itm.back())
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
	}

	template<class T>
	concept NonPaintableStat = std::same_as<T, parse::StatType::CanonicLocal>
	    || std::same_as<T, parse::StatType::CanonicGlobal>;
	template<AnySemOutput Se>
	inline void paintStat(Se& se, const parse::Stat& itm)
	{
		skipSpace(se);
		se.move(itm.place);
		ezmatch(itm.data)(
		    varcase(const NonPaintableStat auto&) {
			    throw std::runtime_error(
			        "Non-paintable statement type found in paintStat: "
			        + std::to_string(itm.data.index()));
		    },
		    varcase(const parse::StatType::Block<Se>&) {
			    paintDoEndBlock(se, var);
		    },
		    varcase(const parse::StatType::ForIn<Se>&) {
			    paintKw<Tok::COND_STAT>(se, "for");
			    paintPat<true, Tok::NAME_TYPE>(se, var.varNames);
			    paintKw<Tok::IN>(se, "in");
			    paintExprOrList(se, var.exprs);

			    paintStatOrRet(se, var.bl);
		    },
		    varcase(const parse::StatType::While<Se>&) {
			    paintKw<Tok::COND_STAT>(se, "while");
			    paintExpr(se, var.cond);
			    paintStatOrRet(se, var.bl);
		    },
		    varcase(const parse::StatType::RepeatUntil<Se>&) {
			    paintKw<Tok::COND_STAT>(se, "repeat");
			    paintStatOrRet(se, var.bl);
			    paintKw<Tok::COND_STAT>(se, "until");
			    paintExpr(se, var.cond);
		    },
		    varcase(const parse::StatType::IfCond<Se>&) {
			    paintIfCond<false>(se, var);
		    },
		    varcase(const parse::StatType::SelfCall&) {
			    paintSelfCall<false>(se, var);
		    },
		    varcase(
		        const parse::StatType::Call&) { paintCall<false>(se, var); },
		    varcase(const parse::StatType::Assign<Se>&) {
			    for (auto& i : var.vars)
				    paintExprData(se, i);
			    paintKw<Tok::ASSIGN>(se, "=");
			    paintExprList(se, var.exprs);
		    },
		    varcase(const parse::StatType::Local<Se>&) {
			    paintVarStat<true>(se, var, "local");
		    },
		    varcase(const parse::StatType::Let<Se>&) {
			    paintVarStat<true>(se, var, "let");
		    },
		    varcase(const parse::StatType::Const<Se>&) {
			    se.pushLocals(var.local2Mp);
			    paintVarStat<false>(se, var, "const");
			    se.popLocals();
		    },

		    varcase(const parse::StatType::Fn&) {
			    paintFunc<false>(se, var, true);
		    },
		    varcase(const parse::StatType::FnDecl<Se>&) {
			    paintFunc<true>(se, var, true);
		    },
		    varcase(const parse::StatType::Function&) {
			    paintFunc<false>(se, var, false);
		    },
		    varcase(const parse::StatType::FunctionDecl<Se>&) {
			    paintFunc<true>(se, var, false);
		    },
		    varcase(const parse::StatType::Semicol) {
			    paintKw<Tok::PUNCTUATION>(se, ";");
		    },
		    varcase(const parse::StatType::Label<Se>&) {
			    paintKw<Tok::PUNCTUATION>(se, ":::");
			    paintName<Tok::NAME_LABEL>(se, var.v);
			    paintKw<Tok::PUNCTUATION>(se, ":");
		    },
		    varcase(const parse::StatType::Goto<Se>&) {
			    paintKw<Tok::COND_STAT>(se, "goto");
			    paintName<Tok::NAME_LABEL>(se, var.v);
		    },

		    // Slu

		    varcase(const parse::StatType::Struct&) {
			    se.pushLocals(var.local2Mp);
			    paintStructBasic(se, var, "struct");
			    paintTable<Tok::NAME_TYPE>(se, var.type);
			    se.popLocals();
		    },
		    varcase(const parse::StatType::Union&) {
			    se.pushLocals(var.local2Mp);
			    paintStructBasic(se, var, "union");
			    paintTable<Tok::NAME_TYPE>(se, var.type);
			    se.popLocals();
		    },

		    varcase(const parse::StatType::Drop<Se>&) {
			    paintKw<Tok::DROP_STAT>(se, "drop");
			    paintExpr(se, var.expr);
		    },
		    varcase(const parse::StatType::Trait&) {
			    paintExportData<Tok::IMPL>(se, var.exported);
			    paintKw<Tok::IMPL>(se, "trait");
			    paintName<Tok::NAME_TRAIT>(se, var.name);

			    skipSpace(se);
			    if (se.in.peek() == '(')
			    {
				    paintKw<Tok::GEN_OP>(se, "(");
				    paintParamList(se, var.params, false);
				    paintKw<Tok::GEN_OP>(se, ")");
			    }
			    if (var.whereSelf.has_value())
			    {
				    paintKw<Tok::GEN_OP>(se, ":");
				    paintTraitExpr(se, *var.whereSelf);
			    }
			    paintWhereClauses(se, var.clauses);
			    paintKw<Tok::BRACES>(se, "{");
			    for (const auto& i : var.itms)
				    paintStat(se, i);
			    paintKw<Tok::BRACES>(se, "}");
		    },
		    varcase(const parse::StatType::Impl&) {
			    paintExportData<Tok::IMPL>(se, var.exported);
			    paintSafety(se,
			        var.isUnsafe ? ast::OptSafety::UNSAFE
			                     : ast::OptSafety::DEFAULT);
			    if (var.deferChecking)
				    paintKw<Tok::IMPL>(se, "defer");
			    paintKw<Tok::IMPL>(se, "impl");

			    skipSpace(se);
			    if (se.in.peek() == '(')
			    {
				    paintKw<Tok::GEN_OP>(se, "(");
				    paintParamList(se, var.params, false);
				    paintKw<Tok::GEN_OP>(se, ")");
			    }

			    if (var.forTrait.has_value())
			    {
				    paintTraitExpr(se, *var.forTrait);
				    paintKw<Tok::IMPL>(se, "for");
			    }
			    paintExpr<Tok::NAME_TYPE>(se, var.type);

			    paintWhereClauses(se, var.clauses);
			    paintKw<Tok::BRACES>(se, "{");
			    for (const auto& i : var.code)
				    paintStat(se, i);
			    paintKw<Tok::BRACES>(se, "}");
		    },
		    varcase(const parse::StatType::ExternBlock<Se>&) {
			    paintSafety(se, var.safety);
			    paintKw<Tok::FN_STAT>(se, "extern");
			    paintString(se, var.abi, var.abiEnd, Tok::FN_STAT);
			    skipSpace(se);
			    bool hadBrace = false;
			    if (se.in.peek() == '{')
			    {
				    hadBrace = true;
				    paintKw<Tok::BRACES>(se, "{");
			    }
			    for (auto& i : var.stats)
				    paintStat(se, i);
			    if (hadBrace)
				    paintKw<Tok::BRACES>(se, "}");
		    },
		    varcase(const parse::StatType::UnsafeBlock<Se>&) {
			    paintKw<Tok::FN_STAT>(se, "unsafe");
			    paintKw<Tok::BRACES>(se, "{");
			    for (auto& i : var.stats)
				    paintStat(se, i);
			    paintKw<Tok::BRACES>(se, "}");
		    },
		    varcase(const parse::StatType::UnsafeLabel) {
			    paintKw<Tok::PUNCTUATION>(se, ":::");
			    paintKw<Tok::FN_STAT>(se, "unsafe");
			    paintKw<Tok::PUNCTUATION>(se, ":");
		    },
		    varcase(const parse::StatType::SafeLabel) {
			    paintKw<Tok::PUNCTUATION>(se, ":::");
			    paintKw<Tok::FN_STAT>(se, "safe");
			    paintKw<Tok::PUNCTUATION>(se, ":");
		    },
		    varcase(const parse::StatType::Use&) {
			    paintExportData<Tok::VAR_STAT>(se, var.exported);
			    paintKw<Tok::VAR_STAT>(se, "use");
			    paintMp<Tok::NAME>(se, var.base);
			    paintUseVariant(se, var.useVariant);
		    },
		    varcase(const parse::StatType::Mod<Se>&) {
			    paintExportData<Tok::CON_STAT>(se, var.exported);
			    paintKw<Tok::CON_STAT>(se, "mod");
			    paintName<Tok::NAME>(se, var.name);
		    },
		    varcase(const parse::StatType::ModAs<Se>&) {
			    paintExportData<Tok::CON_STAT>(se, var.exported);
			    paintKw<Tok::CON_STAT>(se, "mod");
			    paintName<Tok::NAME>(se, var.name);

			    paintKw<Tok::CON_STAT>(se, "as");

			    paintKw<Tok::BRACES>(se, "{");
			    for (auto& i : var.code)
				    paintStat(se, i);
			    paintKw<Tok::BRACES>(se, "}");
		    });
	}
	template<AnySemOutput Se>
	inline void paintExprList(Se& se, const parse::ExprList& itm)
	{
		for (const parse::Expr& i : itm)
		{
			paintExpr(se, i);

			if (&i != &itm.back())
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
	}
	template<AnySemOutput Se>
	inline void paintExprOrList(Se& se, const parse::ExprList& itm)
	{
		return paintExprList(se, itm);
	}
	template<AnySemOutput Se>
	inline void paintExprOrList(Se& se, const parse::Expr& itm)
	{
		return paintExpr(se, itm);
	}
	template<AnySemOutput Se>
	inline void paintBlock(
	    Se& se, const parse::Block<Se>& itm, const bool scopeOwner = true)
	{
		if (scopeOwner)
		{
			skipSpace(se);
			se.move(itm.start);
		}
		for (const parse::Stat& stat : itm.statList)
		{
			paintStat(se, stat);
		}
		if (itm.retTy != parse::RetType::NONE)
		{
			switch (itm.retTy)
			{
			case parse::RetType::RETURN:
				paintKw<Tok::FN_STAT>(se, "return");
				break;
			case parse::RetType::CONTINUE:
				paintKw<Tok::FN_STAT>(se, "continue");
				break;
			case parse::RetType::BREAK:
				paintKw<Tok::FN_STAT>(se, "break");
				break;
			}
			paintExprList(se, itm.retExprs);

			skipSpace(se);
			if (se.in && (se.in.peek() == ';'))
				paintKw<Tok::PUNCTUATION>(se, ";");
		}
	}
	/*
	Make sure to reset in first: `in.reset();`
	*/
	export template<AnySemOutput Se>
	void paintFile(Se& se, const parse::ParsedFile& f)
	{
		for (auto& i : f.code)
			paintStat(se, i);
		skipSpace(se);
	}
} //namespace slu::paint
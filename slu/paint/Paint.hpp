/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/Settings.hpp>
#include <slu/ext/CppMatch.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/State.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/VecInput.hpp>
#include <slu/parser/basic/CharInfo.hpp>
#include <slu/paint/SemOutputStream.hpp>
#include <slu/paint/PaintOps.hpp>
#include <slu/paint/PaintBasics.hpp>

namespace slua::paint
{
	using parse::sluaSyn;

	template<bool SKIP_SPACE = true>
	inline void paintString(AnySemOutput auto& se, const std::string_view sv,const Position end,const Tok tint) 
	{
		if constexpr (SKIP_SPACE)
			skipSpace(se);

		se.template add<Tok::STRING_OUT>(tint == Tok::NONE ? Tok::STRING_OUT : tint);
		const char fChar = se.in.get();
		if (fChar == '[')
		{
			size_t level = 0;
			while (se.in.peek() == '=')
			{
				level++;
				se.template add<Tok::STRING_OUT>(tint == Tok::NONE ? Tok::STRING_OUT : tint);
				se.in.skip();
			}
			se.template add<Tok::STRING_OUT>(tint == Tok::NONE ? Tok::STRING_OUT : tint);
			se.in.skip();

			se.template move<Tok::STRING>(end);
			se.template replPrev<Tok::STRING_OUT>(tint == Tok::NONE ? Tok::STRING_OUT : tint,level+2);
		}
		else
		{
			se.template move<Tok::STRING>(end);
			se.template replPrev<Tok::STRING_OUT>(tint == Tok::NONE ? Tok::STRING_OUT : tint);
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
			se.template add<Tok::NUMBER_KIND>(tint == Tok::NONE ? Tok::NUMBER_KIND : tint,2);
			se.in.skip(2);
		}
		bool wasUscore = false;
		bool parseType = false;
		while (se.in)
		{
			const char chr = se.in.peek();
			if (wasUscore && chr!='_' && (!hex || !parse::isHexDigitChar(chr)) && parse::isValidNameStartChar(chr))
			{
				parseType = true;
				break;
			}
			wasUscore = false;
			if (!hex && (chr == 'e' || chr == 'E')
				|| hex && (chr == 'p' || chr == 'P'))
			{
				se.template add<Tok::NUMBER_KIND>(tint == Tok::NONE ? Tok::NUMBER_KIND : tint, 1);
				se.in.skip();
				const char maybeSign = se.in.peek();
				if (maybeSign == '+' || maybeSign == '-')
				{
					se.template add<Tok::NUMBER_KIND>(tint == Tok::NONE ? Tok::NUMBER_KIND : tint, 1);
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

			if (chr!='.' && chr!='_' && !(hex && parse::isHexDigitChar(chr)) && !parse::isDigitChar(chr))
				break;
			
			se.template add<Tok::NUMBER>(tint == Tok::NONE ? Tok::NUMBER : tint);
			se.in.skip();
		}
		if (parseType)
		{
			while (se.in)
			{
				if (parse::isValidNameChar(se.in.peek()))
				{
					se.template add<Tok::NUMBER_TYPE>(tint == Tok::NONE ? Tok::NUMBER_TYPE : tint);
					se.in.skip();
				}
				else
					break;
			}
		}
	}
	template<AnySemOutput Se>
	inline void paintLifetime(Se& se, const parse::Lifetime& itm)
	{
		for (const auto& i : itm)
		{
			paintKw<Tok::NAME_LIFETIME>(se, "/");
			paintName<Tok::NAME_LIFETIME>(se, i);
		}
	}
	template<AnySemOutput Se>
	inline void paintField(Se& se, const parse::Field<Se>& itm)
	{
		ezmatch(itm)(
		varcase(const parse::FieldType::EXPR2EXPR<Se>&) {
			paintKw<Tok::PUNCTUATION>(se, "[");
			paintExpr(se, var.idx);
			paintKw<Tok::PUNCTUATION>(se, "]");
			paintKw<Tok::ASSIGN>(se, "=");
			paintExpr(se, var.v);
		},
		varcase(const parse::FieldType::NAME2EXPR<Se>&) {
			paintName<Tok::NAME_TABLE>(se, var.idx);
			paintKw<Tok::ASSIGN>(se, "=");
			paintExpr(se, var.v);
		},
		varcase(const parse::FieldType::EXPR<Se>&) {
			paintExpr(se, var.v);
		},
		varcase(const parse::FieldType::NONE) {
			Slua_panic("field shouldnt be FieldType::NONE, found while painting.");
		}
		);
	}
	template<AnySemOutput Se>
	inline void paintTable(Se& se, const parse::TableConstructor<Se>& itm)
	{
		paintKw<Tok::GEN_OP>(se, "{");
		for (const parse::Field<Se>& f : itm)
		{
			paintField(se, f);
			skipSpace(se);
			if (se.in.peek() == ',')
				paintKw<Tok::PUNCTUATION>(se, ",");
			else if (se.in.peek() == ';')
				paintKw<Tok::PUNCTUATION>(se, ";");
		}
		paintKw<Tok::GEN_OP>(se, "}");
	}
	template<AnySemOutput Se>
	inline void paintExpr(Se& se, const parse::Expression<Se>& itm,const Tok tint = Tok::NONE,const bool unOps=true)
	{
		//skipSpace(se);
		se.move(itm.place);
		/*if (std::holds_alternative<parse::ExprType::MULTI_OPERATION>(itm.data))
		{
			//complex version
		}*/
		if (unOps)
		{
			for (const auto& i : itm.unOps)
				paintUnOpItem(se, i);
		}
		ezmatch(itm.data)(
		varcase(const parse::ExprType::MULTI_OPERATION<Se>&) {
			paintExpr(se, *var.first);
			for (const auto& [op,expr] : var.extra)
			{
				paintBinOp(se, op);
				paintExpr(se, expr);
			}
		},
		varcase(const parse::ExprType::FALSE) {
			paintKw<Tok::BUILITIN_VAR>(se, "false");
		},
		varcase(const parse::ExprType::TRUE) {
			paintKw<Tok::BUILITIN_VAR>(se, "true");
		},
		varcase(const parse::ExprType::NIL) {
			paintKw<Tok::BUILITIN_VAR>(se, "nil");
		},
		varcase(const parse::ExprType::VARARGS) {
			paintKw<Tok::PUNCTUATION>(se, "...");
		},
		varcase(const parse::ExprType::OPEN_RANGE) {
			paintKw<Tok::RANGE>(se, "...");
		},
		varcase(const parse::ExprType::LITERAL_STRING&) {
			paintString(se, var.v,var.end,tint);
		},
		varcase(const parse::ExprType::NUMERAL) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::NUMERAL_I128) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::NUMERAL_U128) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::NUMERAL_I64) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::NUMERAL_U64) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::IfCond<Se>&) {
			paintIfCond<true>(se, var);
		},
		varcase(const parse::ExprType::LIFETIME&) {
			paintLifetime(se, var);
		},
		varcase(const parse::ExprType::TYPE_EXPR&) {
			paintTypeExpr(se, var);
		},
		varcase(const parse::ExprType::TRAIT_EXPR&) {
			if constexpr (Se::settings() & sluaSyn)
				paintTraitExpr(se, var);
		},
		varcase(const parse::ExprType::TABLE_CONSTRUCTOR<Se>&) {
			paintTable(se, var.v);
		},
		varcase(const parse::ExprType::LIM_PREFIX_EXP<Se>&) {
			paintLimPrefixExpr(se, *var);
		},
		varcase(const parse::ExprType::FUNCTION_DEF<Se>&) {
			paintFuncDef(se, var.v, parse::MpItmId<Se>({ SIZE_MAX }), false);
		},
		varcase(const parse::ExprType::FUNC_CALL<Se>&) {
			paintLimPrefixExpr(se, *var.val);
			paintArgChain(se, var.argChain);
		},
		varcase(const parse::ExprType::PAT_TYPE_PREFIX&) {
			Slua_panic("Pat type prefix leaked outside of pattern parsing!");
		}
		);
		if constexpr (Se::settings() & sluaSyn)
		{
			if (unOps)
			{
				for (const auto& i : itm.postUnOps)
					paintPostUnOp(se, i);
			}
		}
	}
	template<Tok tok, Tok overlayTok=Tok::NONE,AnySemOutput Se>
	inline void paintMp(Se& se, const parse::MpItmId<Se>& itm)
	{
		const lang::ViewModPath mp = se.in.genData.asVmp(itm);
		for (auto& i : mp)
		{
			if(i=="self")
			{
				paintKw<Tok::VAR_STAT,
					overlayTok == Tok::NONE ? Tok::VAR_STAT : overlayTok
				>(se, "self");
			}
			else if(i=="Self" || i=="crate")
			{
				paintSv<Tok::CON_STAT,
					overlayTok == Tok::NONE ? Tok::CON_STAT : overlayTok
				>(se, i);
			}
			else
			{
				paintSv<tok,
					overlayTok == Tok::NONE ? tok : overlayTok
				>(se, i);
			}

			if (&i != &mp.back())
			{
				paintKw<Tok::MP,
					overlayTok == Tok::NONE ? Tok::MP : overlayTok
				>(se, "::");
			}
		}
	}
	template<AnySemOutput Se>
	inline void paintSubVar(Se& se, const parse::SubVar<Se>& itm)
	{
		paintArgChain(se, itm.funcCalls);

		ezmatch(itm.idx)(
		varcase(const parse::SubVarType::EXPR<Se>&) {
			paintKw<Tok::GEN_OP>(se, "[");
			paintExpr(se, var.idx);
			paintKw<Tok::GEN_OP>(se, "]");
		},
		varcase(const parse::SubVarType::NAME<Se>&) {
			paintKw<Tok::MP_IDX>(se, ".");
			paintName<Tok::NAME>(se, var.idx);
		}
		);
	}
	template<AnySemOutput Se>
	inline void paintVar(Se& se, const parse::Var<Se>& itm)
	{
		ezmatch(itm.base)(
		varcase(const parse::BaseVarType::NAME<Se>&) {
			if constexpr (Se::settings() & sluaSyn)
			{
				if (var.hasDeref)
				{
					paintKw<Tok::DEREF>(se, "*");
					paintMp<Tok::NAME, Tok::DEREF>(se, var.v);
					return;//From match, not func
				}
			}
			paintName<Tok::NAME>(se, var.v);
		},
		varcase(const parse::BaseVarType::EXPR<Se>&) {
			paintKw<Tok::GEN_OP>(se, "(");
			paintExpr(se, var.start);
			paintKw<Tok::GEN_OP>(se, ")");
		},
		varcase(const parse::BaseVarType::EXPR_DEREF_NO_SUB<Se>&) {
			paintKw<Tok::DEREF>(se, "*");
			paintKw<Tok::GEN_OP, Tok::DEREF>(se, "(");
			paintExpr(se, var.start);
			paintKw<Tok::GEN_OP, Tok::DEREF>(se, ")");
		}
		);
		for (const parse::SubVar<Se>& i : itm.sub)
		{
			paintSubVar(se, i);
		}
	}
	template<AnySemOutput Se>
	inline void paintDestrField(Se& se, const parse::DestrField& itm)
	{
		paintKw<Tok::GEN_OP>(se, "|");
		paintName<Tok::NAME>(se, itm.name);
		paintKw<Tok::GEN_OP>(se, "|");
		paintPat(se, itm.pat);
	}
	template<AnySemOutput Se>
	inline void paintTypePrefix(Se& se, const parse::TypePrefix& itm)
	{
		for (const parse::UnOpItem& i : itm)
			paintUnOpItem(se, i);
	}
	template<AnySemOutput Se>
	inline void paintDestrSpec(Se& se, const parse::DestrSpec& itm)
	{
		ezmatch(itm)(
		varcase(const parse::DestrSpecType::Spat&) {
			paintExpr(se, var);
		},
		varcase(const parse::DestrSpecType::Type&) {
			paintTypeExpr(se, var);
		},
		varcase(const parse::DestrSpecType::Prefix&) {
			paintTypePrefix(se, var);
		}
		);
	}
	template<AnySemOutput Se>
	inline void paintPat(Se& se, const parse::Pat& itm)
	{
		ezmatch(itm)(
		varcase(const parse::PatType::Simple&) {
			paintExpr(se, var);
		},
		varcase(const parse::PatType::DestrAny) {
			paintKw<Tok::GEN_OP>(se, "_");
		},
		varcase(const parse::PatType::DestrName&) {
			paintDestrSpec(se, var.spec);
			paintName<Tok::NAME>(se, var.name);
		},
		varcase(const parse::PatType::DestrNameRestrict&) {
			paintDestrSpec(se, var.spec);
			paintName<Tok::NAME>(se, var.name);
			paintKw<Tok::PAT_RESTRICT>(se, "=");
			paintExpr(se, var.restriction);
		},
		
		//parse::PatType::DestrFields or parse::PatType::DestrList
		varcase(const parse::AnyCompoundDestr auto&) 
		{
			paintDestrSpec(se, var.spec);
			paintKw<Tok::GEN_OP>(se, "{");
			for (const auto& i : var.items)
			{
				if constexpr(std::same_as<std::remove_cvref_t<decltype(i)>, parse::DestrField>)
					paintDestrField(se, i);
				else
					paintPat(se, i);

				if (&i != &var.items.back())
					paintKw<Tok::PUNCTUATION>(se, ",");
			}
			if (var.extraFields)
			{
				paintKw<Tok::PUNCTUATION>(se, ",");
				paintKw<Tok::PUNCTUATION>(se, "..");
			}
			paintKw<Tok::GEN_OP>(se, "}");

			paintName<Tok::NAME>(se, var.name);
		}
		);
	}
	template<Tok tok, AnySemOutput Se>
	inline void paintNameList(Se& se, const std::vector<parse::MpItmId<Se>>& itm)
	{
		for (const parse::MpItmId<Se>& i : itm)
		{
			paintName<tok>(se, i);

			if (&i != &itm.back())
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
	}
	template<Tok tok, AnySemOutput Se>
	inline void paintAttribNameList(Se& se, const parse::AttribNameList<Se>& itm)
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
	inline void paintPatOrNamelist(Se& se, const auto& itm)
	{
		if constexpr (Se::settings() & sluaSyn)
			paintPat(se, itm);
		else
		{
			if constexpr (std::same_as<std::remove_cvref_t<decltype(itm)>, parse::MpItmId<Se>>)
				paintName<Tok::NAME>(se, itm);
			else if constexpr (std::same_as<std::remove_cvref_t<decltype(itm)>, parse::AttribNameList<Se>>)
				paintAttribNameList<Tok::NAME>(se, itm);
			else
				paintNameList<Tok::NAME>(se, itm);
		}
	}
	template<AnySemOutput Se>
	inline void paintVarList(Se& se, const std::vector<parse::Var<Se>>& itm)
	{
		for (const parse::Var<Se>& i : itm)
		{
			paintVar(se, i);

			if (&i != &itm.back())
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
	}
	template<AnySemOutput Se>
	inline void paintLimPrefixExpr(Se& se, const parse::LimPrefixExpr<Se>& itm)
	{
		ezmatch(itm)(
		varcase(const parse::LimPrefixExprType::VAR<Se>&) {
			paintVar(se, var.v);
		},
		varcase(const parse::LimPrefixExprType::EXPR<Se>&) {
			paintKw<Tok::GEN_OP>(se, "(");
			paintExpr(se, var.v);
			paintKw<Tok::GEN_OP>(se, ")");
		}
		);
	}
	template<AnySemOutput Se>
	inline void paintArgChain(Se& se, const std::vector<parse::ArgFuncCall<Se>>& itm)
	{
		for (const parse::ArgFuncCall<Se>& i : itm)
		{
			if (!i.funcName.empty())
			{
				paintKw<Tok::MP_IDX>(se, ":");
				paintName<Tok::NAME>(se, i.funcName);
			}
			ezmatch(i.args)(
			varcase(const parse::ArgsType::EXPLIST<Se>&) {
				paintKw<Tok::GEN_OP>(se, "(");
				paintExprList(se, var.v);
				paintKw<Tok::GEN_OP>(se, ")");
			},
			varcase(const parse::ArgsType::TABLE<Se>&) {
				paintTable(se, var.v);
			},
			varcase(const parse::ArgsType::LITERAL&) {
				paintString(se, var.v, var.end, Tok::NONE);
			}
			);
		}
	}
	template<bool forCond,AnySemOutput Se>
	inline void paintEndBlock(Se& se, const parse::Block<Se>& itm)
	{
		if constexpr (Se::settings() & sluaSyn)
		{
			paintBlock(se, itm);
			skipSpace(se);
			paintKw<Tok::BRACES>(se, "}");
		}
		else
		{
			paintBlock(se, itm);
			paintKw<forCond? Tok::COND_STAT : Tok::END_STAT>(se, "end");
		}
	}
	template<AnySemOutput Se>
	inline void paintDoEndBlock(Se& se, const parse::Block<Se>& itm)
	{
		if constexpr (Se::settings() & sluaSyn)
			paintKw<Tok::BRACES>(se, "{");
		else
			paintKw<Tok::COND_STAT>(se, "do");
		paintEndBlock<true>(se, itm);
	}
	template<AnySemOutput Se>
	inline void paintTraitExpr(Se& se, const parse::TraitExpr& itm)
	{
		skipSpace(se);
		se.move(itm.place);
		for (const parse::TraitExprItem& i : itm.traitCombo)
		{
			ezmatch(i)(
			varcase(const parse::TraitExprItemType::FUNC_CALL&) {
				paintLimPrefixExpr(se, *var.val);
				paintArgChain(se, var.argChain);
			},
			varcase(const parse::TraitExprItemType::LIM_PREFIX_EXP&) {
				paintLimPrefixExpr(se, *var);
			}
			);
			if (&i != &itm.traitCombo.back())
				paintKw<Tok::ADD>(se, "+");
		}
	}
	template<AnySemOutput Se>
	inline void paintTypeExpr(Se& se, const parse::TypeExpr& itm, const Tok tint = Tok::NONE)
	{
		if constexpr (Se::settings() & sluaSyn)
		{
			skipSpace(se);
			se.move(itm.place);
			if(itm.hasMut)
				paintKw<Tok::MUT>(se, "mut");
			for (const parse::UnOpItem& i : itm.unOps)
				paintUnOpItem(se, i);

			ezmatch(itm.data)(
			varcase(const parse::TypeExprDataType::ERR_INFERR&) {
				paintKw<Tok::GEN_OP>(se, "?");
			},
			varcase(const parse::TypeExprDataType::ERR&) {
				paintKw<Tok::GEN_OP>(se, "//");
				paintTypeExpr(se, *var.err, tint);
			},
			varcase(const parse::ExprType::NUMERAL_I128) {
				paintNumber(se, tint);
			},
			varcase(const parse::ExprType::NUMERAL_U128) {
				paintNumber(se, tint);
			},
			varcase(const parse::ExprType::NUMERAL_I64) {
				paintNumber(se, tint);
			},
			varcase(const parse::ExprType::NUMERAL_U64) {
				paintNumber(se, tint);
			},
			varcase(const parse::TypeExprDataType::FUNC_CALL&) {
				paintLimPrefixExpr(se, *var.val);
				paintArgChain(se, var.argChain);
			},
			varcase(const parse::TypeExprDataType::LIM_PREFIX_EXP&) {
				paintLimPrefixExpr(se, *var);
			},
			varcase(const parse::TypeExprDataType::SLICER&) {
				paintKw<Tok::GEN_OP>(se, "[");
				paintExpr(se, *var);
				paintKw<Tok::GEN_OP>(se, "]");
			},
			varcase(const parse::TypeExprDataType::Struct&) {
				skipSpace(se);
				if(se.in.peek()=='s')
					paintKw<Tok::CON_STAT>(se, "struct");
				paintTable(se, var);
			},
			varcase(const parse::TypeExprDataType::Union&) {
				paintKw<Tok::CON_STAT>(se, "union");
				paintTable(se, var.fields);
			},
			varcase(const parse::TypeExprDataType::DYN&) {
				paintKw<Tok::DYN>(se, "dyn");
				paintTraitExpr(se, var.expr);
			},
			varcase(const parse::TypeExprDataType::IMPL&) {
				paintKw<Tok::IMPL>(se, "impl");
				paintTraitExpr(se, var.expr);
			},
			varcase(const parse::TypeExprDataType::FN&) {
				paintSafety(se, var.safety);
				paintKw<Tok::FN_STAT>(se, "fn");
				paintTypeExpr(se, *var.argType);
				paintKw<Tok::GEN_OP>(se, "->");
				paintTypeExpr(se, *var.retType);
			},
			varcase(const parse::TypeExprDataType::TRAIT_TY&) {
				paintKw<Tok::NAME_TYPE>(se, "trait");
			},
			varcase(const parse::TypeExprDataType::MULTI_OP&) {
				paintTypeExpr(se, *var.first);
				for (const auto& [op, expr] : var.extra)
				{
					paintBinOp(se, op);
					paintTypeExpr(se, expr);
				}
			}
			);
		}
	}
	template<AnySemOutput Se>
	inline void paintSafety(Se& se, const parse::OptSafety itm)
	{
		switch (itm)
		{
		case parse::OptSafety::SAFE:
			paintKw<Tok::FN_STAT>(se, "safe");
			break;
		case parse::OptSafety::UNSAFE:
			paintKw<Tok::FN_STAT>(se, "unsafe");
			break;
		case parse::OptSafety::DEFAULT:
		default:
			break;
		}
	}
	template<bool DO_END,AnySemOutput Se>
	inline void paintStatOrRet(Se& se, const parse::Block<Se>& itm)
	{
		if constexpr (Se::settings() & sluaSyn)
		{
			skipSpace(se);
			bool hadBrace = false;
			if (se.in.peek() == '{')
			{
				hadBrace = true;
				paintKw<Tok::BRACES>(se, "{");
			}

			paintBlock(se, itm);

			if(hadBrace)
				paintKw<Tok::BRACES>(se, "}");
			return;
		}
		if constexpr(DO_END)
			paintDoEndBlock(se, itm);
		else
			paintBlock(se, itm);
	}
	template<bool DO_END, AnySemOutput Se>
	inline void paintSoeOrBlock(Se& se, const parse::SoeOrBlock<Se>& itm)
	{
		if constexpr (Se::settings() & sluaSyn)
		{
			ezmatch(itm)(
			varcase(const parse::StatOrExprType::BLOCK<Se>&) {
				paintStatOrRet<DO_END>(se, var);
			},
			varcase(const parse::StatOrExprType::EXPR<Se>&) {
				paintKw<Tok::GEN_OP>(se, "=>");
				paintExpr(se, var);
			}
			);
		}
		else
			paintStatOrRet<DO_END>(se, itm);
	}
	template<AnySemOutput Se>
	inline void paintParamList(Se& se, const parse::ParamList<Se>& itm,const bool hasVarArgParam)
	{
		for (const parse::Parameter<Se>& i : itm)
		{
			paintPatOrNamelist(se, i.name);

			if (&i != &itm.back() || hasVarArgParam)
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
		if (hasVarArgParam)
			paintKw<Tok::PUNCTUATION>(se, "...");
	}
	//Pos must be valid, unless the name is empty
	template<AnySemOutput Se>
	inline void paintFuncDef(Se& se, const parse::Function<Se>& func, const parse::MpItmId<Se> name,const lang::ExportData exported, const Position pos = {},const bool fnKw=false)
	{
		if constexpr (Se::settings() & sluaSyn)
		{
			if (exported)
				paintKw<Tok::FN_STAT, Tok::EX_TINT>(se, "ex");

			paintSafety(se, func.safety);
		}
		if(fnKw)
			paintKw<Tok::FN_STAT>(se, "fn");
		else
			paintKw<Tok::FN_STAT>(se, "function");

		if(!name.empty())
			se.move(pos);

		paintName<Tok::NAME>(se, name);
		paintKw<Tok::GEN_OP>(se, "(");
		paintParamList(se, func.params, func.hasVarArgParam);

		paintKw<Tok::GEN_OP>(se, ")");
		if constexpr (Se::settings() & sluaSyn)
		{
			if (func.retType.has_value())
			{
				paintKw<Tok::GEN_OP>(se, "->");
				paintTypeExpr(se, *func.retType);
			}
			paintKw<Tok::BRACES>(se, "{");
		}
		//No do, for functions in lua
		paintEndBlock<false>(se, func.block);
		
	}
	template<bool isExpr,AnySemOutput Se>
	inline void paintIfCond(Se& se,
		const parse::BaseIfCond<Se,isExpr>& itm
	)
	{
		paintKw<Tok::COND_STAT>(se, "if");
		paintExpr(se, *itm.cond);

		if constexpr (!(Se::settings() & sluaSyn))
			paintKw<Tok::COND_STAT>(se, "then");

		paintSoeOrBlock<false>(se, *itm.bl);
		for (const auto& [cond, bl] : itm.elseIfs)
		{
			if constexpr (Se::settings() & sluaSyn)
			{
				paintKw<Tok::COND_STAT>(se, "else");
				paintKw<Tok::COND_STAT>(se, "if");
				paintExpr(se, cond);
			}
			else
			{
				paintKw<Tok::COND_STAT>(se, "elseif");
				paintExpr(se, cond);
				paintKw<Tok::COND_STAT>(se, "then");
			}
			paintSoeOrBlock<false>(se, bl);
		}
		if (itm.elseBlock.has_value())
		{
			paintKw<Tok::COND_STAT>(se, "else");
			paintSoeOrBlock<false>(se, **itm.elseBlock);
		}

		if constexpr (!(Se::settings() & sluaSyn))
			paintKw<Tok::COND_STAT>(se, "end");
	}
	template<AnySemOutput Se>
	inline void paintUseVariant(Se& se, const parse::UseVariant& itm)
	{
		ezmatch(itm)(
			// use xxxx::Xxx;
		varcase(const parse::UseVariantType::IMPORT&) {},

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
			for (const parse::MpItmId<Se>& i : var)
			{
				paintName<Tok::NAME>(se, i);

				if (&i != &var.back())
					paintKw<Tok::PUNCTUATION>(se, ",");
			}
			paintKw<Tok::BRACES>(se, "}");
		}
		);
	}
	template<size_t TOK_SIZE,AnySemOutput Se>
	inline void paintVarStat(Se& se, const auto& itm, const char(&tokChr)[TOK_SIZE])
	{
		if constexpr (Se::settings() & sluaSyn)
		{
			if (itm.exported)
				paintKw<Tok::VAR_STAT, Tok::EX_TINT>(se, "ex");
		}

		paintKw<Tok::VAR_STAT>(se, tokChr);

		paintPatOrNamelist(se, itm.names);

		if (itm.exprs.empty())return;

		paintKw<Tok::ASSIGN>(se, "=");
		paintExprList(se, itm.exprs);
	}
	template<size_t TOK_SIZE,AnySemOutput Se>
	inline void paintStructBasic(Se& se, const auto& itm, const char(&tokChr)[TOK_SIZE])
	{
		if (itm.exported)
			paintKw<Tok::CON_STAT, Tok::EX_TINT>(se, "ex");

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
	template<AnySemOutput Se>
	inline void paintStat(Se& se, const parse::Statement<Se>& itm)
	{
		skipSpace(se);
		se.move(itm.place);
		ezmatch(itm.data)(
		varcase(const parse::StatementType::BLOCK<Se>&) {
			paintDoEndBlock(se, var.bl);
		},
		varcase(const parse::StatementType::FOR_LOOP_NUMERIC<Se>&) {
			paintKw<Tok::COND_STAT>(se, "for");

			paintPatOrNamelist(se, var.varName);
			paintKw<Tok::ASSIGN>(se, "=");

			paintExpr(se, var.start);
			paintKw<Tok::PUNCTUATION>(se, ",");
			paintExpr(se, var.end);
			if (var.step.has_value())
			{
				paintKw<Tok::PUNCTUATION>(se, ",");
				paintExpr(se, *var.step);
			}
			paintStatOrRet<true>(se, var.bl);
		},
		varcase(const parse::StatementType::FOR_LOOP_GENERIC<Se>&) {
			paintKw<Tok::COND_STAT>(se, "for");

			paintPatOrNamelist(se, var.varNames);
			paintKw<Tok::IN>(se, "in");

			paintExprOrList(se, var.exprs);

			paintStatOrRet<true>(se, var.bl);
		},
		varcase(const parse::StatementType::WHILE_LOOP<Se>&) {
			paintKw<Tok::COND_STAT>(se, "while");

			paintExpr(se, var.cond);

			paintStatOrRet<true>(se, var.bl);
		},
		varcase(const parse::StatementType::REPEAT_UNTIL<Se>&) {
			paintKw<Tok::COND_STAT>(se, "repeat");

			paintStatOrRet<false>(se, var.bl);

			paintKw<Tok::COND_STAT>(se, "until");

			paintExpr(se, var.cond);
		},
		varcase(const parse::StatementType::IfCond<Se>&) {
			paintIfCond<false>(se, var);
		},
		varcase(const parse::StatementType::FUNC_CALL<Se>&) {
			paintLimPrefixExpr(se, *var.val);
			paintArgChain(se, var.argChain);
		},
		varcase(const parse::StatementType::ASSIGN<Se>&) {
			paintVarList(se, var.vars);
			paintKw<Tok::ASSIGN>(se, "=");
			paintExprList(se, var.exprs);
		},
		varcase(const parse::StatementType::LOCAL_ASSIGN<Se>&) {
			paintVarStat(se,var, "local");
		},
		varcase(const parse::StatementType::LET<Se>&) {
			paintVarStat(se,var, "let");
		},
		varcase(const parse::StatementType::CONST<Se>&) {
			paintVarStat(se,var, "const");
		},
		varcase(const parse::StatementType::FN<Se>&) {
			if constexpr (Se::settings() & sluaSyn)
				paintFuncDef(se, var.func, var.name, var.exported, var.place, true);
			else
				paintFuncDef(se, var.func, var.name, false, var.place, true);
		},
		varcase(const parse::StatementType::FUNCTION_DEF<Se>&) {
			if constexpr (Se::settings() & sluaSyn)
				paintFuncDef(se, var.func, var.name, var.exported, var.place);
			else
				paintFuncDef(se, var.func, var.name, false, var.place);
		},
		varcase(const parse::StatementType::LOCAL_FUNCTION_DEF<Se>&) {
			paintKw<Tok::FN_STAT>(se, "local");
			paintFuncDef(se, var.func, var.name, false, var.place);
		},
		varcase(const parse::StatementType::BREAK) {
			paintKw<Tok::COND_STAT>(se, "break");
		},
		varcase(const parse::StatementType::SEMICOLON) {
			paintKw<Tok::PUNCTUATION>(se, ";");
		},
		varcase(const parse::StatementType::LABEL<Se>&) {
			paintKw<Tok::PUNCTUATION>(se, parse::sel<Se>("::", ":::"));
			paintName<Tok::NAME_LABEL>(se, var.v);
			paintKw<Tok::PUNCTUATION>(se, parse::sel<Se>("::", ":"));
		},
		varcase(const parse::StatementType::GOTO<Se>&) {
			paintKw<Tok::COND_STAT>(se, "goto");
			paintName<Tok::NAME_LABEL>(se, var.v);
		},

		// Slua

		varcase(const parse::StatementType::Struct<Se>&) {
			paintStructBasic(se, var, "struct");

			skipSpace(se);
			if(se.in.peek()=='=')
				paintKw<Tok::ASSIGN>(se, "=");
			paintTypeExpr(se, var.type);
		},
		varcase(const parse::StatementType::Union<Se>&) {
			paintStructBasic(se, var, "union");
			paintTable(se, var.type);
		},

		varcase(const parse::StatementType::DROP<Se>&) {
			paintKw<Tok::DROP_STAT>(se, "drop");
			paintExpr(se, var.expr);
		},
		varcase(const parse::StatementType::UNSAFE_LABEL) {
			paintKw<Tok::PUNCTUATION>(se, ":::");
			paintKw<Tok::FN_STAT>(se, "unsafe");
			paintKw<Tok::PUNCTUATION>(se, ":");
		},
		varcase(const parse::StatementType::SAFE_LABEL) {
			paintKw<Tok::PUNCTUATION>(se, ":::");
			paintKw<Tok::FN_STAT>(se, "safe");
			paintKw<Tok::PUNCTUATION>(se, ":");
		},
		varcase(const parse::StatementType::MOD_DEF<Se>&) {
			if (var.exported)
				paintKw<Tok::CON_STAT, Tok::EX_TINT>(se, "ex");
			paintKw<Tok::CON_STAT>(se, "mod");
			paintName<Tok::NAME>(se, var.name);
		},
		varcase(const parse::StatementType::USE&) {
			if constexpr (Se::settings() & sluaSyn)
			{
				if (var.exported)
					paintKw<Tok::VAR_STAT, Tok::EX_TINT>(se, "ex");

				paintKw<Tok::VAR_STAT>(se, "use");
				paintMp<Tok::NAME>(se, var.base);
				paintUseVariant(se, var.useVariant);
			}
		},
		varcase(const parse::StatementType::MOD_DEF_INLINE<Se>&) {
			if (var.exported)
				paintKw<Tok::CON_STAT, Tok::EX_TINT>(se, "ex");
			paintKw<Tok::CON_STAT>(se, "mod");
			paintName<Tok::NAME>(se, var.name);

			paintKw<Tok::CON_STAT>(se, "as");

			paintKw<Tok::BRACES>(se, "{");
			paintBlock(se, var.bl);
			paintKw<Tok::BRACES>(se, "}");
		}
		);
	}
	template<AnySemOutput Se>
	inline void paintExprList(Se& se, const parse::ExpList<Se>& itm)
	{
		for (const parse::Expression<Se>& i : itm)
		{
			paintExpr(se, i);

			if (&i != &itm.back())
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
	}
	template<AnySemOutput Se>
	inline void paintExprOrList(Se& se, const parse::ExpList<Se>& itm) {
		return paintExprList(se, itm);
	}
	template<AnySemOutput Se>
	inline void paintExprOrList(Se& se, const parse::Expression<Se>& itm) {
		return paintExpr(se, itm);
	}
	template<AnySemOutput Se>
	inline void paintBlock(Se& se, const parse::Block<Se>& itm)
	{
		skipSpace(se);
		se.move(itm.start);
		for (const parse::Statement<Se>& stat : itm.statList)
		{
			paintStat(se, stat);
		}
		if (itm.hadReturn)
		{
			if constexpr(Se::settings()&sluaSyn)
			{
				skipSpace(se);
				if (se.in.peek() == 'd')
					paintKw<Tok::FN_STAT, false>(se, "do");
			}
			paintKw<Tok::FN_STAT>(se,"return");
			paintExprList(se, itm.retExprs);

			skipSpace(se);
			if (se.in && (se.in.peek() == ';'))
				paintKw<Tok::PUNCTUATION>(se, ";");
		}
	}
	/*
	Make sure to reset in first: `in.reset();`
	*/
	template<AnySemOutput Se>
	inline void paintFile(Se& se, const parse::ParsedFile<Se>& f) {
		paintBlock(se, f.code);
		skipSpace(se);
	}
}
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

namespace slu::paint
{
	template<bool SKIP_SPACE = true>
	inline void paintString(AnySemOutput auto& se, const std::string_view sv,const Position end,const Tok tint) 
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
			se.template replPrev<Tok::STRING_OUT>(tint,level+2);
		}
		else
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
			se.template add<Tok::NUMBER_KIND>(tint,2);
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

			if (chr!='.' && chr!='_' && !(hex && parse::isHexDigitChar(chr)) && !parse::isDigitChar(chr))
				break;
			
			se.template add<Tok::NUMBER>(tint);
			se.in.skip();
		}
		if (parseType)
		{
			while (se.in)
			{
				if (parse::isValidNameChar(se.in.peek()))
				{
					se.template add<Tok::NUMBER_TYPE>(tint);
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
			paintKw<Tok::LIFETIME_SEP>(se, "/");
			paintName<Tok::NAME_LIFETIME>(se, i);
		}
	}
	template<Tok nameTok,AnySemOutput Se>
	inline void paintField(Se& se, const parse::Field<Se>& itm)
	{
		ezmatch(itm)(
		varcase(const parse::FieldType::Expr2Expr<Se>&) {
			paintKw<Tok::PUNCTUATION>(se, "[");
			paintExpr(se, var.idx);
			paintKw<Tok::PUNCTUATION>(se, "]");
			paintKw<Tok::ASSIGN>(se, "=");
			paintExpr<nameTok>(se, var.v);
		},
		varcase(const parse::FieldType::Name2Expr<Se>&) {
			paintName<Tok::NAME_TABLE>(se, var.idx);
			paintKw<Tok::ASSIGN>(se, "=");
			paintExpr<nameTok>(se, var.v);
		},
		varcase(const parse::FieldType::Expr<Se>&) {
			paintExpr<nameTok>(se, var);
		},
		varcase(const parse::FieldType::NONE) {
			Slu_panic("field shouldnt be FieldType::NONE, found while painting.");
		}
		);
	}
	template<Tok nameTok,AnySemOutput Se>
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
	inline void paintTypeExpr(Se& se, const parse::Expr<Se>& itm, const Tok tint = Tok::NONE, const bool unOps = true) {
		paintExpr<Tok::NAME_TYPE>(se, itm, tint, unOps);
	}
	template<Tok nameTok=Tok::NAME,AnySemOutput Se>
	inline void paintExpr(Se& se, const parse::Expr<Se>& itm,const Tok tint = Tok::NONE,const bool unOps=true)
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
	template<Tok nameTok=Tok::NAME,AnySemOutput Se>
	inline void paintExprData(Se& se, const parse::ExprData<Se>& itm,const Tok tint = Tok::NONE)
	{
		ezmatch(itm)(
		varcase(const parse::ExprType::MpRoot) {
			paintKw<Tok::MP_ROOT>(se, ":>");
		},
		varcase(const parse::ExprType::Local) {
			paintNameOrLocal<true,nameTok>(se, var);
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
		varcase(const parse::ExprType::Index<Se>&) {
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
		varcase(const parse::ExprType::SelfCall<Se>&) {
			paintSelfCall<true>(se, var);
		},
		varcase(const parse::ExprType::Call<Se>&) {
			paintCall<true>(se, var);
		},
		varcase(const parse::ExprType::MultiOp<Se>&) {
			paintExpr<nameTok>(se, *var.first);
			for (const auto& [op,expr] : var.extra)
			{
				paintBinOp(se, op);
				paintExpr<nameTok>(se, expr);
			}
		},
		varcase(const parse::ExprType::False) {
			paintKw<Tok::BUILITIN_VAR>(se, "false");
		},
		varcase(const parse::ExprType::True) {
			paintKw<Tok::BUILITIN_VAR>(se, "true");
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
			paintString(se, var.v,var.end,tint);
		},
		varcase(const parse::ExprType::F64) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::P128) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::M128) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::I64) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::U64) {
			paintNumber(se, tint);
		},
		varcase(const parse::ExprType::IfCond<Se>&) {
			paintIfCond<true>(se, var);
		},
		varcase(const parse::ExprType::Lifetime&) {
			paintLifetime(se, var);
		},
		varcase(const parse::ExprType::TraitExpr&) {
			paintTraitExpr(se, var);
		},
		varcase(const parse::ExprType::Table<Se>&) {
			paintTable<nameTok>(se, var);
		},
		varcase(const parse::ExprType::Function<Se>&) {
			paintFuncDef(se, var, parse::MpItmId<Se>::newEmpty(), false);
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
		varcase(const parse::ExprType::Slice&) {
			paintKw<Tok::GEN_OP>(se, "[");
			paintTypeExpr(se, *var.v);
			paintKw<Tok::GEN_OP>(se, "]");
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
		}
		);
	}
	template<Tok tok, Tok overlayTok=Tok::NONE,AnySemOutput Se>
	inline void paintMp(Se& se, const parse::MpItmId<Se>& itm)
	{
		skipSpace(se);
		if(parse::checkToken(se.in,":>"))
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
				if(parse::checkTextToken(se.in,mp[i]))
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
	inline void paintDestrField(Se& se, const parse::DestrField<Se, isLocal>& itm)
	{
		paintKw<Tok::GEN_OP>(se, "|");
		paintPoolStr<Tok::NAME_TABLE>(se, itm.name);
		paintKw<Tok::GEN_OP>(se, "|");
		paintPat<isLocal,nameTok>(se, itm.pat);
	}
	template<AnySemOutput Se>
	inline void paintTypePrefix(Se& se, const parse::TypePrefix& itm)
	{
		for (const parse::UnOpItem& i : itm)
			paintUnOpItem(se, i);
	}
	template<Tok nameTok,AnySemOutput Se>
	inline void paintDestrSpec(Se& se, const parse::DestrSpec<Se>& itm)
	{
		ezmatch(itm)(
		varcase(const parse::DestrSpecType::Spat<Se>&) {
			paintExpr<nameTok>(se, var);
		},
		varcase(const parse::DestrSpecType::Prefix&) {
			paintTypePrefix(se, var);
		}
		);
	}
	template<bool isLocal,Tok nameTok,AnySemOutput Se>
	inline void paintPat(Se& se, const parse::Pat<Se, isLocal>& itm)
	{
		ezmatch(itm)(

			//parse::PatType::DestrFields or parse::PatType::DestrList
			varcase(const auto&) //requires(parse::AnyCompoundDestr<isLocal, decltype(var)>)
		{
			paintDestrSpec<nameTok>(se, var.spec);
			paintKw<Tok::GEN_OP>(se, "{");
			for (const auto& i : var.items)
			{
				if constexpr (std::same_as<std::remove_cvref_t<decltype(i)>, parse::DestrFieldV<true, isLocal>>)
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
		varcase(const parse::PatType::Simple<Se>&) {
			paintExpr(se, var);
		},
		varcase(const parse::PatType::DestrAny<Se, isLocal>) {
			paintKw<Tok::GEN_OP>(se, "_");
		},
		varcase(const parse::PatType::DestrName<Se,isLocal>&) {
			paintDestrSpec<nameTok>(se, var.spec);
			paintNameOrLocal<isLocal,Tok::NAME>(se, var.name);
		},
		varcase(const parse::PatType::DestrNameRestrict<Se, isLocal>&) {
			paintDestrSpec<nameTok>(se, var.spec);
			paintNameOrLocal<isLocal,Tok::NAME>(se, var.name);
			paintKw<Tok::PAT_RESTRICT>(se, "=");
			paintExpr(se, var.restriction);
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
	inline void paintArgs(Se& se, const parse::Args<Se>& itm)
	{
		ezmatch(itm)(
			varcase(const parse::ArgsType::ExprList<Se>&) {
			paintKw<Tok::GEN_OP>(se, "(");
			paintExprList(se, var);
			paintKw<Tok::GEN_OP>(se, ")");
		},
			varcase(const parse::ArgsType::Table<Se>&) {
			paintTable<Tok::NAME>(se, var);
		},
			varcase(const parse::ArgsType::String&) {
			paintString(se, var.v, var.end, Tok::NONE);
		}
		);
	}
	template<bool forCond,AnySemOutput Se>
	inline void paintEndBlock(Se& se, const parse::Block<Se>& itm, const bool scopeOwner = true)
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
		for (const parse::ExprV<true>& i : itm.traitCombo)
		{
			paintExpr<Tok::NAME_TRAIT>(se, i);
			if (&i != &itm.traitCombo.back())
				paintKw<Tok::ADD>(se, "+");
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

		paintBlock(se, itm);

		if (hadBrace)
			paintKw<Tok::BRACES>(se, "}");
		return;
	}
	template<AnySemOutput Se>
	inline void paintSoeOrBlock(Se& se, const parse::SoeOrBlock<Se>& itm)
	{
		ezmatch(itm)(
		varcase(const parse::SoeType::Block<Se>&) {
			paintStatOrRet(se, var);
		},
		varcase(const parse::SoeType::Expr<Se>&) {
			paintKw<Tok::GEN_OP>(se, "=>");
			paintExpr(se, *var);
		}
		);
	}
	template<bool isLocal,AnySemOutput Se>
	inline void paintParamList(Se& se, const parse::ParamList<isLocal>& itm,const bool hasVarArgParam)
	{
		for (const parse::Parameter<isLocal>& i : itm)
		{
			paintNameOrLocal<isLocal>(se, i.name);
			paintKw<Tok::PAT_RESTRICT>(se, "=");
			paintTypeExpr(se, i.type);
			

			if (&i != &itm.back() || hasVarArgParam)
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
		if (hasVarArgParam)
			paintKw<Tok::PUNCTUATION>(se, "...");
	}
	template<Tok baseCol,AnySemOutput Se>
	inline void paintExportData(Se& se, parse::ExportData exported)
	{
		if (exported)
			paintKw<baseCol, Tok::EX_TINT>(se, "ex");
	}
	//Pos must be valid, unless the name is empty
	template<AnySemOutput Se>
	inline void paintFuncDecl(Se& se, const parse::ParamList<true>& params,const bool hasVarArgParam, const std::optional<std::unique_ptr<parse::ExprV<true>>>& retType, const parse::MpItmId<Se> name, const lang::ExportData exported,const parse::OptSafety safety, const Position pos = {}, const bool fnKw = false)
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
		paintParamList(se, params, hasVarArgParam);
		paintKw<Tok::GEN_OP>(se, ")");

		if (retType.has_value())
		{
			paintKw<Tok::GEN_OP>(se, "->");
			paintTypeExpr(se, **retType);
		}
	}
	template<bool isDecl,AnySemOutput Se>
	inline void paintFunc(Se& se, const auto& itm, const bool fnKw)
	{
		if constexpr (isDecl)
		{
			se.pushLocals(itm.local2Mp);
			paintFuncDecl(se, itm.params, itm.hasVarArgParam, itm.retType,
				itm.name, itm.exported, itm.safety, itm.place, fnKw);
			se.popLocals();
		} else {
			paintFuncDef(se, itm.func, itm.name, itm.exported, itm.place, fnKw);
		}
	}
	//Pos must be valid, unless the name is empty
	template<AnySemOutput Se>
	inline void paintFuncDef(Se& se, const parse::Function<Se>& func, const parse::MpItmId<Se> name,const lang::ExportData exported, const Position pos = {},const bool fnKw=false)
	{
		std::optional<std::unique_ptr<parse::ExprV<true>>> emptyTy{};
		const std::optional<std::unique_ptr<parse::ExprV<true>>>* retType;
		parse::OptSafety safety;
		retType = &func.retType;
		safety = func.safety;
		se.pushLocals(func.local2Mp);

		paintFuncDecl(se, func.params, func.hasVarArgParam,*retType, name, exported, safety, pos, fnKw);
		paintKw<Tok::BRACES>(se, "{");

		//No do, for functions in lua
		paintEndBlock<false>(se, func.block,false);

		se.popLocals();
	}
	template<bool isExpr,AnySemOutput Se>
	inline void paintIfCond(Se& se,
		const parse::BaseIfCond<Se,isExpr>& itm
	)
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
				paintMp<Tok::NAME>(se, i);

				if (&i != &var.back())
					paintKw<Tok::PUNCTUATION>(se, ",");
			}
			paintKw<Tok::BRACES>(se, "}");
		}
		);
	}
	template<bool isLocal,size_t TOK_SIZE,AnySemOutput Se>
	inline void paintVarStat(Se& se, const auto& itm, const char(&tokChr)[TOK_SIZE])
	{
		paintExportData<Tok::VAR_STAT>(se,itm.exported);

		paintKw<Tok::VAR_STAT>(se, tokChr);

		paintPat<isLocal,Tok::NAME_TYPE>(se, itm.names);

		if (itm.exprs.empty())return;

		paintKw<Tok::ASSIGN>(se, "=");
		paintExprList(se, itm.exprs);
	}
	template<size_t TOK_SIZE,AnySemOutput Se>
	inline void paintStructBasic(Se& se, const auto& itm, const char(&tokChr)[TOK_SIZE])
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
	inline void paintCall(Se& se, const parse::Call<Se,boxed>& itm)
	{
		paintExpr(se, *itm.v);
		paintArgs(se, itm.args);
	}
	template<bool boxed, AnySemOutput Se>
	inline void paintSelfCall(Se& se, const parse::SelfCall<Se,boxed>& itm)
	{
		paintExpr(se, *itm.v);
		paintKw<Tok::GEN_OP>(se, ".");
		paintName(se, itm.method);
		paintArgs(se, itm.args);
	}

	template<AnySemOutput Se>
	inline void paintWhereClauses(Se& se, const parse::WhereClauses& itm)
	{
		if (itm.empty())return;
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
	concept NonPaintableStat = std::same_as<T,parse::StatementType::CanonicLocal>
		|| std::same_as<T, parse::StatementType::CanonicGlobal>;
	template<AnySemOutput Se>
	inline void paintStat(Se& se, const parse::Statement<Se>& itm)
	{
		skipSpace(se);
		se.move(itm.place);
		ezmatch(itm.data)(
		varcase(const NonPaintableStat auto&) {
			throw std::runtime_error(
				"Non-paintable statement type found in paintStat: " + std::to_string(itm.data.index())
			);
		},
		varcase(const parse::StatementType::Block<Se>&) {
			paintDoEndBlock(se, var);
		},
		varcase(const parse::StatementType::ForIn<Se>&) {
			paintKw<Tok::COND_STAT>(se, "for");
			paintPat<true,Tok::NAME_TYPE>(se, var.varNames);
			paintKw<Tok::IN>(se, "in");
			paintExprOrList(se, var.exprs);

			paintStatOrRet(se, var.bl);
		},
		varcase(const parse::StatementType::While<Se>&) {
			paintKw<Tok::COND_STAT>(se, "while");
			paintExpr(se, var.cond);
			paintStatOrRet(se, var.bl);
		},
		varcase(const parse::StatementType::RepeatUntil<Se>&) {
			paintKw<Tok::COND_STAT>(se, "repeat");
			paintStatOrRet(se, var.bl);
			paintKw<Tok::COND_STAT>(se, "until");
			paintExpr(se, var.cond);
		},
		varcase(const parse::StatementType::IfCond<Se>&) {
			paintIfCond<false>(se, var);
		},
		varcase(const parse::StatementType::SelfCall<Se>&) {
			paintSelfCall<false>(se, var);
		},
		varcase(const parse::StatementType::Call<Se>&) {
			paintCall<false>(se, var);
		},
		varcase(const parse::StatementType::Assign<Se>&) {
			for (auto& i : var.vars)
				paintExprData(se, i);
			paintKw<Tok::ASSIGN>(se, "=");
			paintExprList(se, var.exprs);
		},
		varcase(const parse::StatementType::Local<Se>&) {
			paintVarStat<true>(se,var, "local");
		},
		varcase(const parse::StatementType::Let<Se>&) {
			paintVarStat<true>(se,var, "let");
		},
		varcase(const parse::StatementType::Const<Se>&) {
			se.pushLocals(var.local2Mp);
			paintVarStat<false>(se,var, "const");
			se.popLocals();
		},

		varcase(const parse::StatementType::Fn<Se>&) {
			paintFunc<false>(se, var, true);
		},
		varcase(const parse::StatementType::FnDecl<Se>&) {
			paintFunc<true>(se, var, true);
		},
		varcase(const parse::StatementType::Function<Se>&) {
			paintFunc<false>(se, var, false);
		},
		varcase(const parse::StatementType::FunctionDecl<Se>&) {
			paintFunc<true>(se, var, false);
		},
		varcase(const parse::StatementType::LocalFunctionDef<Se>&) {
			paintKw<Tok::FN_STAT>(se, "local");
			paintFuncDef(se, var.func, var.name, false, var.place);
		},
		varcase(const parse::StatementType::Semicol) {
			paintKw<Tok::PUNCTUATION>(se, ";");
		},
		varcase(const parse::StatementType::Label<Se>&) {
			paintKw<Tok::PUNCTUATION>(se, ":::");
			paintName<Tok::NAME_LABEL>(se, var.v);
			paintKw<Tok::PUNCTUATION>(se, ":");
		},
		varcase(const parse::StatementType::Goto<Se>&) {
			paintKw<Tok::COND_STAT>(se, "goto");
			paintName<Tok::NAME_LABEL>(se, var.v);
		},

		// Slu

		varcase(const parse::StatementType::Struct&) {
			se.pushLocals(var.local2Mp);
			paintStructBasic(se, var, "struct");
			paintTable<Tok::NAME_TYPE>(se, var.type);
			se.popLocals();
		},
		varcase(const parse::StatementType::Union&) {
			se.pushLocals(var.local2Mp);
			paintStructBasic(se, var, "union");
			paintTable<Tok::NAME_TYPE>(se, var.type);
			se.popLocals();
		},

		varcase(const parse::StatementType::Drop<Se>&) {
			paintKw<Tok::DROP_STAT>(se, "drop");
			paintExpr(se, var.expr);
		},
		varcase(const parse::StatementType::Trait&) {
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
		varcase(const parse::StatementType::Impl&) {
			paintExportData<Tok::IMPL>(se, var.exported);
			paintSafety(se, var.isUnsafe ? parse::OptSafety::UNSAFE : parse::OptSafety::DEFAULT);
			if (var.deferChecking) paintKw<Tok::IMPL>(se, "defer");
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
		varcase(const parse::StatementType::ExternBlock<Se>&) {
			paintSafety(se, var.safety);
			paintKw<Tok::FN_STAT>(se, "extern");
			paintString(se, var.abi,var.abiEnd,Tok::FN_STAT);
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
		varcase(const parse::StatementType::UnsafeBlock<Se>&) {
			paintKw<Tok::FN_STAT>(se, "unsafe");
			paintKw<Tok::BRACES>(se, "{");
			for (auto& i : var.stats)
				paintStat(se, i);
			paintKw<Tok::BRACES>(se, "}");
		},
		varcase(const parse::StatementType::UnsafeLabel) {
			paintKw<Tok::PUNCTUATION>(se, ":::");
			paintKw<Tok::FN_STAT>(se, "unsafe");
			paintKw<Tok::PUNCTUATION>(se, ":");
		},
		varcase(const parse::StatementType::SafeLabel) {
			paintKw<Tok::PUNCTUATION>(se, ":::");
			paintKw<Tok::FN_STAT>(se, "safe");
			paintKw<Tok::PUNCTUATION>(se, ":");
		},
		varcase(const parse::StatementType::Use&) {
			paintExportData<Tok::VAR_STAT>(se, var.exported);
			paintKw<Tok::VAR_STAT>(se, "use");
			paintMp<Tok::NAME>(se, var.base);
			paintUseVariant(se, var.useVariant);
		},
		varcase(const parse::StatementType::Mod<Se>&) {
			paintExportData<Tok::CON_STAT>(se, var.exported);
			paintKw<Tok::CON_STAT>(se, "mod");
			paintName<Tok::NAME>(se, var.name);
		},
		varcase(const parse::StatementType::ModAs<Se>&) {
			paintExportData<Tok::CON_STAT>(se, var.exported);
			paintKw<Tok::CON_STAT>(se, "mod");
			paintName<Tok::NAME>(se, var.name);

			paintKw<Tok::CON_STAT>(se, "as");

			paintKw<Tok::BRACES>(se, "{");
			for (auto& i : var.code)
				paintStat(se, i);
			paintKw<Tok::BRACES>(se, "}");
		}
		);
	}
	template<AnySemOutput Se>
	inline void paintExprList(Se& se, const parse::ExprList<Se>& itm)
	{
		for (const parse::Expr<Se>& i : itm)
		{
			paintExpr(se, i);

			if (&i != &itm.back())
				paintKw<Tok::PUNCTUATION>(se, ",");
		}
	}
	template<AnySemOutput Se>
	inline void paintExprOrList(Se& se, const parse::ExprList<Se>& itm) {
		return paintExprList(se, itm);
	}
	template<AnySemOutput Se>
	inline void paintExprOrList(Se& se, const parse::Expr<Se>& itm) {
		return paintExpr(se, itm);
	}
	template<AnySemOutput Se>
	inline void paintBlock(Se& se, const parse::Block<Se>& itm,const bool scopeOwner = true)
	{
		if(scopeOwner)
		{
			skipSpace(se);
			se.move(itm.start);
		}
		for (const parse::Statement<Se>& stat : itm.statList)
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
	template<AnySemOutput Se>
	inline void paintFile(Se& se, const parse::ParsedFile<Se>& f) {
		for (auto& i : f.code)
			paintStat(se, i);
		skipSpace(se);
	}
}
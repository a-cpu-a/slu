/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>
#include <optional>
#include <memory>
#include <variant>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/ext/CppMatch.hpp>
#include <slu/ext/ExtendVariant.hpp>
#include <slu/lang/BasicState.hpp>
#include <slu/BigInt.hpp>
#include "Input.hpp"

namespace slu::parse
{
	template<AnyCfgable CfgT, template<bool> class T>
	using SelV = T<CfgT::settings()& sluSyn>;

	template<bool isSlu, class T, class SlT>
	using Sel = std::conditional_t<isSlu, SlT, T>;

#define Slu_DEF_CFG(_Name) template<AnyCfgable CfgT> using _Name = SelV<CfgT, _Name ## V>
#define Slu_DEF_CFG2(_Name,_ArgName) template<AnyCfgable CfgT,bool _ArgName> using _Name =Sel<CfgT::settings()& sluSyn, _Name ## V<false, _ArgName>, _Name ## V<true, _ArgName>>
#define Slu_DEF_CFG_CAPS(_NAME) template<AnyCfgable CfgT> using _NAME = SelV<CfgT, _NAME ## v>

	template<AnyCfgable Cfg, size_t TOK_SIZE, size_t TOK_SIZE2>
	consteval const auto& sel(const char(&tok)[TOK_SIZE], const char(&sluTok)[TOK_SIZE2])
	{
		if constexpr (Cfg::settings() & sluSyn)
			return sluTok;
		else
			return tok;
	}
	template<bool boxed, class T>
	struct MayBox
	{
		Sel<boxed, T, std::unique_ptr<T>> v;

		T& get() {
			if constexpr (boxed) return *v; else return v;
		}
		const T& get() const {
			if constexpr (boxed) return *v; else return v;
		}

		T& operator*() { return get(); }
		const T& operator*() const { return get(); }

		T& operator->() { return &get(); }
		const T* operator->() const { return &get(); }
	};
	template<bool boxed, class T>
	constexpr auto mayBoxFrom(T&& v)
	{
		if constexpr (boxed)
			return MayBox<true, T>(std::make_unique<T>(std::move(v)));
		else
			return MayBox<false, T>(std::move(v));
	}
	template<class T>
	constexpr MayBox<false, T> wontBox(T&& v) {
		return MayBox<false, T>(std::move(v));
	}

	//Mp ref
	template<AnyCfgable CfgT> using MpItmId = SelV<CfgT, lang::MpItmIdV>;



	//Forward declare

	template<bool isSlu> struct StatementV;
	Slu_DEF_CFG(Statement);

	template<bool isSlu> struct ExprV;
	Slu_DEF_CFG(Expr);
	template<bool isSlu>
	using BoxExprV = std::unique_ptr<ExprV<isSlu>>;

	template<bool isSlu> struct VarV;
	Slu_DEF_CFG(Var);

	namespace FieldType
	{
		//For lua only!
		template<bool isSlu> struct Expr2ExprV;
		Slu_DEF_CFG(Expr2Expr);

		template<bool isSlu> struct Name2ExprV;
		Slu_DEF_CFG(Name2Expr);

		using parse::ExprV;
		using parse::Expr;
	}
	namespace LimPrefixExprType
	{
		template<bool isSlu> struct VARv;
		Slu_DEF_CFG_CAPS(VAR);

		using parse::ExprV;
		using parse::Expr;
	}
	template<bool isSlu>
	using LimPrefixExprV = std::variant<
		LimPrefixExprType::VARv<isSlu>,
		LimPrefixExprType::ExprV<isSlu>
	>;
	Slu_DEF_CFG(LimPrefixExpr);

	template<bool isSlu> struct ArgFuncCallV;
	Slu_DEF_CFG(ArgFuncCall);

	template<bool isSlu> struct FuncCallV;
	Slu_DEF_CFG(FuncCall);

	template<bool isSlu>
	using ExprListV = std::vector<ExprV<isSlu>>;
	Slu_DEF_CFG(ExprList);

	namespace ExprType
	{
		struct String { std::string v; Position end; };	// "LiteralString"	

		// "Numeral"
		using F64 = double;
		using I64 = int64_t;

		//u64,i128,u128, for slu only
		using U64 = uint64_t;
		using P128 = Integer128<false>;
		using M128 = Integer128<false, true>;
	}

	// Slu

	using slu::lang::MpItmIdV;
	using slu::lang::ModPath;
	using slu::lang::ModPathView;
	using slu::lang::ExportData;
	using SubModPath = std::vector<std::string>;


	using PoolString = lang::LocalObjId;//implicitly unknown mp.

}
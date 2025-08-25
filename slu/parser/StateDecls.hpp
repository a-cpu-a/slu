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
import slu.big_int;
#include "Input.hpp"

namespace slu::parse
{
	using lang::MpItmId;

	template<bool flag, class FalseT, class TrueT>
	using Sel = std::conditional_t<flag, TrueT,FalseT>;

#define Slu_DEF_CFG(_Name) template<class CfgT> using _Name = _Name ## V<true>
#define Slu_DEF_CFG2(_Name,_ArgName) template<class CfgT,bool _ArgName> using _Name =_Name ## V<true, _ArgName>

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

		T* operator->() { return &get(); }
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

	//Forward declare
	struct Stat;
	struct Expr;
	using BoxExpr = std::unique_ptr<Expr>;

	namespace FieldType
	{
		//For lua only! (currently)
		struct Expr2Expr;
		struct Name2Expr;
		using parse::Expr;
	}

	template<bool isSlu>
	using ExprListV = std::vector<Expr>;
	Slu_DEF_CFG(ExprList);

	namespace ExprType
	{
		struct OpenRange {};
		struct String { std::string v; Position end; };

		// "Numeral"
		using F64 = double;
		using I64 = int64_t;

		//u64,i128,u128, for slu only
		using U64 = uint64_t;
		using P128 = Integer128<false>;
		using M128 = Integer128<false, true>;
	}

	using slu::lang::ModPath;
	using slu::lang::ModPathView;
	using slu::lang::ExportData;
	using SubModPath = std::vector<std::string>;

	using PoolString = lang::LocalObjId;//implicitly unknown mp.
}
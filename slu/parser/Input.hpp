﻿/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <format>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/Settings.hpp>
#include <slu/lang/BasicState.hpp>
#include <slu/Ansi.hpp>
#include "Position.hpp"

namespace slu::parse
{
	template<class T, bool isSlu>
	concept AnyGenDataV =
#ifdef Slu_NoConcepts
		true
#else
		requires(T t,lang::MpItmIdV<isSlu> v) {
			{ t.asSv(v) } -> std::same_as<std::string_view>;
			{ t.resolveEmpty() } -> std::same_as<lang::MpItmIdV<isSlu>>;

			{ t.resolveUnknown(std::string()) } -> std::same_as<lang::MpItmIdV<isSlu>>;
			{ t.resolveUnknown(lang::ModPath()) } -> std::same_as<lang::MpItmIdV<isSlu>>;

			//{ t.resolveNameOrLocal(std::string()) } -> std::same_as<parse::DynLocalOrNameV<isSlu>>;
			{ t.resolveName(std::string()) } -> std::same_as<lang::MpItmIdV<isSlu>>;
			{ t.resolveName(lang::ModPath()) } -> std::same_as<lang::MpItmIdV<isSlu>>;
	}
#endif // Slu_NoConcepts
	;
	/*
	template<class T>
	concept AnyGenData =
#ifdef Slu_NoConcepts
		true
#else
		AnyGenDataV<T,true> || AnyGenDataV<T, false>
#endif // Slu_NoConcepts
	;*/
	
	//Here, so streamed inputs can be made
	template<class T>
	concept AnyInput =
#ifdef Slu_NoConcepts
		true
#else

		AnyCfgable<T> && requires(T t) {

		//{ t.genData } -> AnyGenData;
		{ t.genData } -> AnyGenDataV<std::remove_cvref_t<T>::settings()&sluSyn>;

		{ t.restart() } -> std::same_as<void>;

		{ t.skip() } -> std::same_as<void>;
		{ t.skip((size_t)100) } -> std::same_as<void>;

		{ t.get() } -> std::same_as<uint8_t>;
		{ t.get((size_t)100) } -> std::same_as<std::span<const uint8_t>>;

		{ t.peek() } -> std::same_as<uint8_t>;
		{ t.peekAt((size_t)100) } -> std::same_as<uint8_t>;
		{ t.peek((size_t)100) } -> std::same_as<std::span<const uint8_t>>;


		/* Returns true, while stream still has stuff */
		//{ (bool)t } -> std::same_as<bool>; //Crashes intelisense


		{ t.isOob((size_t)100) } -> std::same_as<bool>;


		//Error output

		{ t.fileName() } -> std::same_as<std::string_view>;
		{ t.getLoc() } -> std::same_as<Position>;

		//Management
		{ t.newLine() } -> std::same_as<void>;

		{t.handleError(std::string()) } -> std::same_as<void>;
		{t.hasError() } -> std::same_as<bool>;
	}
#endif // Slu_NoConcepts
	;

	inline std::string errorLocStr(const AnyInput auto& in,const Position pos) {
		return std::format(
			" {}:" 
			LUACC_NUM_COL("{}")
			":" 
			LUACC_NUM_COL("{}"),

			in.fileName(),
			pos.line, pos.index+1
		);
			//" " + in.fileName() + "(" LUACC_NUMBER + std::to_string(pos.line) + LUACC_DEFAULT "):" LUACC_NUMBER + std::to_string(pos.index);
	}
	inline std::string errorLocStr(const AnyInput auto& in) {
		return errorLocStr(in,in.getLoc());
	}

	struct EndOfStreamError : std::exception
	{
		std::string m;
		EndOfStreamError(const AnyInput auto& in) :m(std::format("Unexpected end of stream.{}",errorLocStr(in))) {}
		const char* what() const { return m.c_str(); }
	};
}
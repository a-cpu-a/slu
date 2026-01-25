/*
    A program file.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This file is part of Slu-c.

    Slu-c is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Slu-c is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with Slu-c.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;

#include <format>
#include <span>
#include <string>

#include <slu/Ansi.hpp>
export module slu.parse.input;

import slu.settings;
import slu.ast.pos;
import slu.lang.basic_state;

namespace slu::parse
{
	export template<class T, bool isSlu>
	concept AnyGenDataV =
#ifdef Slu_NoConcepts
	    true
#else
	    requires(T t, lang::MpItmId v) {
		    { t.asSv(v) } -> std::same_as<std::string_view>;
		    { t.resolveEmpty() } -> std::same_as<lang::MpItmId>;

		    { t.resolveUnknown(std::string()) } -> std::same_as<lang::MpItmId>;
		    {
			    t.resolveUnknown(lang::ModPath())
		    } -> std::same_as<lang::MpItmId>;

		    //{ t.resolveNameOrLocal(std::string()) } ->
		    //std::same_as<parse::DynLocalOrNameV<isSlu>>;
		    { t.resolveName(std::string()) } -> std::same_as<lang::MpItmId>;
		    { t.resolveName(lang::ModPath()) } -> std::same_as<lang::MpItmId>;
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
	export template<class T>
	concept AnyInput =
#ifdef Slu_NoConcepts
	    true
#else

		AnyCfgable<T> && requires(T t) {

		//{ t.genData } -> AnyGenData;
		{ t.genData } -> AnyGenDataV<true>;

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
		{ t.getLoc() } -> std::same_as<ast::Position>;

		//Management
		{ t.newLine() } -> std::same_as<void>;

		{t.handleError(std::string()) } -> std::same_as<void>;
		{t.hasError() } -> std::same_as<bool>;
	}
#endif // Slu_NoConcepts
	    ;

	export std::string errorLocStr(
	    const AnyInput auto& in, const ast::Position pos)
	{
		return std::format(" {}:" LUACC_NUM_COL("{}") ":" LUACC_NUM_COL("{}"),

		    in.fileName(), pos.line, pos.index + 1);
		//" " + in.fileName() + "(" LUACC_NUMBER + std::to_string(pos.line) +
		//LUACC_DEFAULT "):" LUACC_NUMBER + std::to_string(pos.index);
	}
	export std::string errorLocStr(const AnyInput auto& in)
	{
		return errorLocStr(in, in.getLoc());
	}

	export struct EndOfStreamError : std::exception
	{
		std::string m;
		EndOfStreamError(const AnyInput auto& in)
		    : m(std::format("Unexpected end of stream.{}", errorLocStr(in)))
		{}
		const char* what() const noexcept override
		{
			return m.c_str();
		}
	};
} //namespace slu::parse
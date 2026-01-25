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
#include <cstdint>
#include <ranges>
#include <span>
#include <string>
#include <vector>

export module slu.paint.sem_output;
import slu.settings;
import slu.ast.pos;
import slu.ast.state;
import slu.parse.input;
import slu.parse.manage_newl;

namespace slu::paint
{
	export enum class Tok : uint8_t
	{
		NONE = 0,

		WHITESPACE = 0,    //comments too
		COMMENT_OUTER,     //The -- [=[ ]=] parts
		COMMENT_WIP,       //tod..., fixm...
		COMMENT_DOC,       // text inside ---
		COMMENT_DOC_OUTER, // ---

		COND_STAT, //if, else, elseif, for, while, try, match, do
		VAR_STAT,  //let, local, const, use, self

		//fn, function, extern, safe, unsafe, async, return, do return, throw,
		//break, continue
		FN_STAT,

		CON_STAT, //struct, enum, union, mod, axiom, impl, trait, crate, Self

		END_STAT, //end

		DROP_STAT,    //drop
		BUILITIN_VAR, //false true nil

		//ex (overlay over export, which has the same color as next token)
		EX_TINT,

		//(overlay over comptime, which has the same color as next token)
		COMP_TINT,

		ASSIGN,       // =
		PAT_RESTRICT, // =
		GEN_OP,       // =>, ==, != || () {} [] -> _
		PUNCTUATION,  // >= <= > < , ; :::_:
		MP_IDX,       // . : ::
		MP,           // ::
		MP_ROOT,      // :>
		BRACES, // {} (may be combined with some tint, to get a colored brace)

		NAME,       // .123 .xxx xxx
		NAME_TABLE, // xxx inside {xxx=...} or |xxx| ... patterns
		NAME_TYPE,  // type trait xxx
		NAME_TRAIT, // xxx
		NAME_LABEL,
		NAME_LIFETIME,
		LIFETIME_SEP,

		NUMBER,
		NUMBER_TYPE, // u8, u16, ...
		NUMBER_KIND, // 0x, 0b, 0o, eE+- pP+-
		STRING,
		STRING_OUT, // String, the quotes, [=['s or the type (`c"xxx"`, the c)

		SPLICER,    // @()
		SPLICE_VAR, // $xxx xxx$

		ANNOTATION, // @xxx @xxx{}

		AS,
		IN,

		IMPL,
		DYN,
		MUT,

		TRY, // ? operator

		REP,

		OR,
		AND,

		PAT_OR,
		PAT_AND,

		ADD,

		MOD,
		DIV, //floor div too
		MUL,

		BIT_OR,
		BIT_XOR,
		BIT_AND,

		NOT,
		SUB,
		NEG,

		CONCAT,
		RANGE,
		SHL,
		SHR,

		DEREF,
		EXP,

		REF,
		REF_SHARE,
		REF_MUT,
		REF_CONST,

		PTR,
		PTR_SHARE,
		PTR_MUT,
		PTR_CONST,

		ENUM_COUNT
	};

	//Here, so custom semantic token outputs can be made
	export template<class T>
	concept AnySemOutput =
#ifdef Slu_NoConcepts
	    true
#else
		parse::AnyCfgable<T> && requires(T t) {

		//{ t.db } -> std::same_as<LuaMpDb>;

			{ t.in } -> parse::AnyInput;
			{ t.out } -> std::same_as<std::vector<std::vector<typename T::SemPair>>&>;

			{ t.popLocals() } -> std::same_as<void>;

			{ t.move(ast::Position()) } -> std::same_as<T&>;
			{ t.template move<Tok::WHITESPACE>(ast::Position()) } -> std::same_as<T&>;

			{ t.add(Tok::WHITESPACE) } -> std::same_as<T&>;
			{ t.add(Tok::WHITESPACE,Tok::WHITESPACE) } -> std::same_as<T&>;
			{ t.template add<Tok::WHITESPACE>() } -> std::same_as<T&>;
			{ t.template add<Tok::WHITESPACE>(Tok::WHITESPACE) } -> std::same_as<T&>;
			{ t.template add<Tok::WHITESPACE, Tok::WHITESPACE>() } -> std::same_as<T&>;
			{ t.add(Tok::WHITESPACE,1ULL) } -> std::same_as<T&>;
			{ t.add(Tok::WHITESPACE,Tok::WHITESPACE, 1ULL) } -> std::same_as<T&>;
			{ t.template add<Tok::WHITESPACE>(1ULL) } -> std::same_as<T&>;
			{ t.template add<Tok::WHITESPACE>(Tok::WHITESPACE,1ULL) } -> std::same_as<T&>;
			{ t.template add<Tok::WHITESPACE, Tok::WHITESPACE>(1ULL) } -> std::same_as<T&>;

			{ t.replPrev(Tok::WHITESPACE) } -> std::same_as<T&>;
			{ t.replPrev(Tok::WHITESPACE,Tok::WHITESPACE) } -> std::same_as<T&>;
			{ t.template replPrev<Tok::WHITESPACE>() } -> std::same_as<T&>;
			{ t.template replPrev<Tok::WHITESPACE>(Tok::WHITESPACE) } -> std::same_as<T&>;
			{ t.template replPrev<Tok::WHITESPACE, Tok::WHITESPACE>() } -> std::same_as<T&>;
			{ t.replPrev(Tok::WHITESPACE, 1ULL) } -> std::same_as<T&>;
			{ t.replPrev(Tok::WHITESPACE,Tok::WHITESPACE, 1ULL) } -> std::same_as<T&>;
			{ t.template replPrev<Tok::WHITESPACE>(1ULL) } -> std::same_as<T&>;
			{ t.template replPrev<Tok::WHITESPACE>(Tok::WHITESPACE, 1ULL) } -> std::same_as<T&>;
			{ t.template replPrev<Tok::WHITESPACE, Tok::WHITESPACE>(1ULL) } -> std::same_as<T&>;

	}
#endif // Slu_NoConcepts
	    ;

	export template<class Converter>
	constexpr uint32_t basicTokBlend(const Tok tok, const Tok overlayTok)
	{
		const uint32_t from = Converter::tok2Col(tok);
		if (overlayTok == Tok::NONE)
			return from; //no overlay, just return the color
		const uint32_t to = Converter::tok2Col(overlayTok);

		const uint8_t t = 100;
		const uint8_t s = 0xFF - t;
		return ((((((from >> 0) & 0xff) * s + ((to >> 0) & 0xff) * t) >> 8))
		    | (((((from >> 8) & 0xff) * s + ((to >> 8) & 0xff) * t)) & ~0xff)
		    | (((((from >> 16) & 0xff) * s + ((to >> 16) & 0xff) * t) << 8)
		        & ~0xffff)
		    | (((((from >> 24) & 0xff) * s + ((to >> 24) & 0xff) * t) << 16)
		        & ~0xffffff));
	}

	export struct ColorConverter
	{
		static constexpr uint32_t tok2Col(const Tok tok)
		{
			switch (tok)
			{
				//Add colors:
			case Tok::WHITESPACE:
			default:
				return (0x122311 * uint32_t(tok)
				           ^ 0x388831 * (uint32_t(tok) + 1))
				    | 0xFF000000;
			}
		}
		static constexpr uint32_t from(const Tok tok, const Tok overlayTok)
		{
			return basicTokBlend<ColorConverter>(tok, overlayTok);
		}
	};

	//Converter::from(Tok,Tok) -> SemPair
	export template<parse::AnyInput In, class Converter = ColorConverter>
	struct SemOutput
	{
		//Possible a color, likely u16 or u32
		using SemPair
		    = decltype(Converter::from(Tok::WHITESPACE, Tok::WHITESPACE));

		constexpr SemOutput(In& in) : in(in) {}

		constexpr static auto settings()
		{
			return In::settings();
		}

		In& in;
		std::vector<std::vector<SemPair>> out;
		std::vector<const parse::Locals<In>*> localStack;

		auto resolveLocal(parse::LocalId local)
		{
			return localStack.back()->names.at(local.v);
		}
		void pushLocals(const parse::Locals<In>& locals)
		{
			localStack.push_back(&locals);
		}
		void popLocals()
		{
			localStack.pop_back();
		}

		SemOutput& addRaw(const SemPair p, size_t count = 1)
		{
			ast::Position loc = in.getLoc();
			//in.skip(count);
			if (out.size() <= loc.line - 1)
				out.resize(loc.line);
			out[loc.line - 1].resize(loc.index + count, p);
			return *this;
		}
		template<Tok t, Tok overlayTok> SemOutput& add(size_t count = 1)
		{
			return addRaw(Converter::from(t, overlayTok), count);
		}
		template<Tok t> SemOutput& add(size_t count = 1)
		{
			return add<t, t>(count);
		}
		template<Tok t> SemOutput& add(Tok overlayTok, size_t count = 1)
		{
			return addRaw(Converter::from(t, overlayTok), count);
		}
		SemOutput& add(Tok t, Tok overlayTok, size_t count = 1)
		{
			return addRaw(Converter::from(t, overlayTok), count);
		}
		SemOutput& add(Tok t, size_t count = 1)
		{
			return add(t, t, count);
		}

		SemOutput& replPrevRaw(const SemPair p, size_t count = 1)
		{
			for (auto& k : std::ranges::reverse_view{out})
			{
				for (auto& pv : std::ranges::reverse_view{k})
				{
					pv = p;
					count--;
					if (count == 0)
						return *this;
				}
			}
			return *this;
		}
		template<Tok t, Tok overlayTok> SemOutput& replPrev(size_t count = 1)
		{
			return replPrevRaw(Converter::from(t, overlayTok), count);
		}
		template<Tok t> SemOutput& replPrev(size_t count = 1)
		{
			return replPrev<t, t>(count);
		}
		template<Tok t> SemOutput& replPrev(Tok overlayTok, size_t count = 1)
		{
			return replPrevRaw(Converter::from(t, overlayTok), count);
		}
		SemOutput& replPrev(Tok t, Tok overlayTok, size_t count = 1)
		{
			return replPrevRaw(Converter::from(t, overlayTok), count);
		}
		SemOutput& replPrev(Tok t, size_t count = 1)
		{
			return replPrev(t, t, count);
		}

		template<Tok t> SemOutput& move(ast::Position p)
		{
			parse::ParseNewlineState nlState = parse::ParseNewlineState::NONE;

			constexpr SemPair pair = Converter::tok2Col(t);

			// handle newlines, while moving towards 'p'
			while (in)
			{
				ast::Position loc = in.getLoc();

				if (loc.index == p.index && loc.line == p.line)
					break;

				//Already skipping
				if (parse::manageNewlineState<false>(in.get(), nlState, in))
				{
					out.emplace_back();
					continue;
				}

				//Add the color to the char.
				if (out.size() <= loc.line - 1)
					out.resize(loc.line);
				out[loc.line - 1].resize(loc.index + 1, pair);
			}
			return *this;
		}
		SemOutput& move(ast::Position p)
		{
			return move<Tok::WHITESPACE>(p);
		}
	};
} //namespace slu::paint
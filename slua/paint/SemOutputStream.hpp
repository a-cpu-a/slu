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

#include <slua/Settings.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/State.hpp>
#include <slua/parser/adv/SkipSpace.hpp>

namespace slua::paint
{
	using parse::AnyCfgable;
	using parse::AnyInput;
	using parse::AnySettings;
	using parse::Setting;

	using parse::Position;

	//ms sad
#undef IN

	enum class Tok : uint8_t
	{
		NONE = 0,

		WHITESPACE = 0, //comments too
		COMMENT_OUTER, //The -- [=[ ]=] parts
		COMMENT_WIP, //tod..., fixm...
		COMMENT_DOC, // text inside ---
		COMMENT_DOC_OUTER, // ---

		COND_STAT,//if, else, elseif, for, while, try, match, do
		VAR_STAT,//let, local, const, use, self
		FN_STAT,//fn, function, extern, safe, unsafe, async, return, do return, throw, break, continue
		CON_STAT,//struct, enum, union, mod, axiom, impl, trait, crate, Self

		DROP_STAT,//drop
		BUILITIN_VAR,//false true nil

		EX_TINT,//ex (overlay over export, which has the same color as next token)
		COMP_TINT,//comptime (overlay over comptime, which has the same color as next token)
		TYPE_TAG, //type

		END_STAT,//end

		ASSIGN,// =
		PAT_RESTRICT,// =
		GEN_OP,// =>, ==, != || () {} [] -> _
		PUNCTUATION,// >= <= > < , ;
		MP,// ::
		BRACES,// {} (may be combined with some tint, to get a colored brace)

		MP_IDX,// . : ::
		NAME,// .123 .xxx xxx
		NAME_TABLE,// xxx inside {xxx=...}
		NAME_TYPE,// type trait xxx
		NAME_LABEL,
		NAME_LIFETIME,

		NUMBER,
		NUMBER_TYPE,// u8, u16, ...
		NUMBER_KIND,// 0x, 0b, 0o, eE pP
		STRING,
		STRING_OUT, // String, the quotes, [=['s or the type (`c"xxx"`, the c)

		SPLICER,// @()
		SPLICE_VAR,// $xxx xxx$

		ANNOTATION,// @xxx @xxx{}


		AS,
		IN,

		IMPL,
		DYN,
		MUT,

		TRY,// ? operator

		ARRAY_CONSTRUCT,

		OR,
		AND,

		PAT_OR,
		PAT_AND,

		ADD,

		MOD,
		DIV,//floor div too
		MUL,

		BIT_OR,
		BIT_XOR,
		BIT_AND,

		NOT,
		SUB,
		NEG,

		SIZEOF,// lua # un operator
		BITNOT,// lua ~ un operator

		CONCAT,
		RANGE,
		SHIFT,

		DEREF,
		EXP,

		REF,
		REF_SHARE,

		PTR_CONST,
	};

	//Here, so custom semantic token outputs can be made
	template<class T>
	concept AnySemOutput =
#ifdef Slua_NoConcepts
		true
#else
		AnyCfgable<T> && requires(T t) {

		//{ t.db } -> std::same_as<LuaMpDb>;

			{ t.in } -> AnyInput;
			{ t.out } -> std::same_as<std::vector<std::vector<typename T::SemPair>>&>;
			//Hack for skip space: temporary!!!
			{ t.commentPair() } -> std::same_as<typename T::SemPair>;

			{ t.move(Position()) } -> std::same_as<T&>;
			{ t.template move<Tok::WHITESPACE>(Position()) } -> std::same_as<T&>;

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
#endif // Slua_NoConcepts
	;

	template<class Converter>
	constexpr uint32_t basicTokBlend(const Tok tok, const Tok overlayTok)
	{
		const uint32_t to = Converter::tok2Col(tok);
		const uint32_t from = Converter::tok2Col(overlayTok);

		const uint8_t t = 100;
		const uint8_t s = 0xFF - t;
		return (
			(((((from >> 0) & 0xff) * s +
				((to >> 0) & 0xff) * t) >> 8)) |
			(((((from >> 8) & 0xff) * s +
				((to >> 8) & 0xff) * t)) & ~0xff) |
			(((((from >> 16) & 0xff) * s +
				((to >> 16) & 0xff) * t) << 8) & ~0xffff) |
			(((((from >> 24) & 0xff) * s +
				((to >> 24) & 0xff) * t) << 16) & ~0xffffff)
			);
	}

	struct ColorConverter
	{
		static constexpr uint32_t tok2Col(const Tok tok)
		{
			switch (tok)
			{
				//Add colors:
			case Tok::WHITESPACE:
			default:
				return (0x122311 * uint32_t(tok) ^ 0x388831 * (uint32_t(tok) + 1)) | 0xFF000000;
			}
		}
		static constexpr uint32_t from(const Tok tok, const Tok overlayTok) {
			return basicTokBlend<ColorConverter>(tok, overlayTok);
		}
	};

	//Converter::from(Tok,Tok) -> SemPair
	template<AnyInput In,class Converter= ColorConverter>
	struct SemOutput
	{
		//Possible a color, likely u16 or u32
		using SemPair = decltype(Converter::from(Tok::WHITESPACE, Tok::WHITESPACE));
		
		constexpr SemOutput(In& in) : in(in) {}

		constexpr static auto settings()
		{
			return In::settings();
		}

		consteval static SemPair commentPair() {
			return Converter::from(Tok::WHITESPACE, Tok::WHITESPACE);
		}

		In& in;
		std::vector<std::vector<SemPair>> out;

		SemOutput& addRaw(const SemPair p, size_t count = 1)
		{
			Position loc = in.getLoc();
			//in.skip(count);
			if (out.size() <= loc.line-1)
				out.resize(loc.line);
			out[loc.line-1].resize(loc.index + count,p);
			return *this;
		}
		template<Tok t,Tok overlayTok>
		SemOutput& add(size_t count = 1) {
			return addRaw(Converter::from(t,overlayTok), count);
		}
		template<Tok t>
		SemOutput& add(size_t count = 1) {
			return add<t,t>(count);
		}
		template<Tok t>
		SemOutput& add(Tok overlayTok,size_t count = 1) {
			return addRaw(Converter::from(t, overlayTok), count);
		}
		SemOutput& add(Tok t,Tok overlayTok, size_t count = 1) {
			return addRaw(Converter::from(t, overlayTok), count);
		}
		SemOutput& add(Tok t,size_t count=1) {
			return add(t,t, count);
		}

		SemOutput& replPrevRaw(const SemPair p, size_t count = 1)
		{
			for (auto& k : std::ranges::reverse_view{ out })
			{
				for (auto& pv : std::ranges::reverse_view{ k })
				{
					pv = p;
					count--;
					if(count==0)
						return *this;
				}
			}
			return *this;
		}
		template<Tok t,Tok overlayTok>
		SemOutput& replPrev(size_t count = 1) {
			return replPrevRaw(Converter::from(t,overlayTok), count);
		}
		template<Tok t>
		SemOutput& replPrev(size_t count = 1) {
			return replPrev<t,t>(count);
		}
		template<Tok t>
		SemOutput& replPrev(Tok overlayTok, size_t count = 1) {
			return replPrevRaw(Converter::from(t, overlayTok), count);
		}
		SemOutput& replPrev(Tok t,Tok overlayTok, size_t count = 1) {
			return replPrevRaw(Converter::from(t, overlayTok), count);
		}
		SemOutput& replPrev(Tok t, size_t count = 1) {
			return replPrev(t,t, count);
		}

		template<Tok t>
		SemOutput& move(Position p) 
		{
			parse::ParseNewlineState nlState = parse::ParseNewlineState::NONE;

			constexpr SemPair pair = Converter::tok2Col(t);

			// handle newlines, while moving towards 'p'
			while (in)
			{
				Position loc = in.getLoc();

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
		SemOutput& move(Position p) {
			return move<Tok::WHITESPACE>(p);
		}
	};
}
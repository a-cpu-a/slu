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

namespace slua::paint
{
	using parse::AnyCfgable;
	using parse::AnyInput;
	using parse::AnySettings;
	using parse::Setting;

	using parse::Position;

	enum class Tok : uint8_t
	{
		WHITESPACE = 0, //comments too
		COMMENT_OUTER, //The -- [=[ ]=] parts
		COMMENT_WIP, //tod..., fixm...
		COMMENT_DOC, // text inside ---
		COMMENT_DOC_OUTER, // ---

		COND_STAT,//if, else, elseif, for, while, try, match, do
		VAR_STAT,//let, local, const, use
		FN_STAT,//fn, function, extern, safe, unsafe
		RET_STAT,//return, do return, throw, break, continue
		CON_STAT,//struct, enum, union, mod, axiom, impl, trait

		DROP_STAT,//drop, as

		EX_TINT,//ex (overlay over export, which has the same color as next token)
		COMP_TINT,//comptime (overlay over comptime, which has the same color as next token)

		END_STAT,//end

		ASSIGN,// =
		PAT_RESTRICT,// =
		GEN_OP,// =>, ==, != || () {} []

		MP_IDX,// : ::
		NAME,// .123 .xxx xxx
		NAME_TABLE,// xxx inside {xxx=...}
		NAME_TYPE,

		NUMBER,
		NUMBER_KIND,
		STRING,
		STRING_OUT, // the quotes, [=['s or the type (`c"xxx"`, the c)

		SPLICER,// @()
		SPLICE_VAR,// $xxx xxx$

		ANNOTATION,// @xxx @xxx{}


		IMPL,
		DYN,
		MUT,

		TRY,// ? operator

		OR,
		AND,

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

			{ t.move(Position()) } -> std::same_as<T&>;

			{ t.add(Tok::WHITESPACE) } -> std::same_as<T&>;
			{ t.add(Tok::WHITESPACE,Tok::WHITESPACE) } -> std::same_as<T&>;
			{ t.template add<Tok::WHITESPACE>() } -> std::same_as<T&>;
			{ t.template add<Tok::WHITESPACE, Tok::WHITESPACE>() } -> std::same_as<T&>;

			{ t.add(Tok::WHITESPACE,1ULL) } -> std::same_as<T&>;
			{ t.add(Tok::WHITESPACE,Tok::WHITESPACE, 1ULL) } -> std::same_as<T&>;
			{ t.template add<Tok::WHITESPACE>(1ULL) } -> std::same_as<T&>;
			{ t.template add<Tok::WHITESPACE, Tok::WHITESPACE>(1ULL) } -> std::same_as<T&>;

	}
#endif // Slua_NoConcepts
	;

	struct ColorConverter
	{
		static constexpr uint32_t tok2Col(const Tok tok)
		{
			switch (tok)
			{
				//Add colors:
			case Tok::WHITESPACE:
			default:
				return 0x388831 * (uint32_t(tok) + 1) | 0xFF000000;
			}
		}
		static constexpr uint32_t from(const Tok tok, const Tok overlayTok)
		{
			const uint32_t to = tok2Col(tok);
			const uint32_t from = tok2Col(overlayTok);

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
	};

	//Converter::from(Tok,Tok) -> SemPair
	template<AnyInput In,class Converter= ColorConverter,AnySettings SettingsT = Setting<void>>
	struct SemOutput
	{
		using SemPair = decltype(Converter::from(Tok::WHITESPACE, Tok::WHITESPACE));
		
		constexpr SemOutput(In& in, SettingsT) :in(in) {}
		constexpr SemOutput(In& in) : in(in) {}

		constexpr static SettingsT settings()
		{
			return SettingsT();
		}

		In& in;
		std::vector<std::vector<SemPair>> out;

		SemOutput& addRaw(const SemPair p, size_t count = 1)
		{
			//TODO
			return *this;
		}
		template<Tok t,Tok overlayTok>
		SemOutput& add(size_t count = 1) {
			return addRaw(Converter::from(t,overlayTok), count));
		}
		template<Tok t>
		SemOutput& add(size_t count = 1) {
			return add<t,t>(count);
		}
		SemOutput& add(Tok t,Tok overlayTok, size_t count = 1) {
			return addRaw(Converter::from(t, overlayTok), count));
		}
		SemOutput& add(Tok t,size_t count=1) {
			return add(t,t, count));
		}
		SemOutput& move(Position p) {

			return *this;
		}
	};
}
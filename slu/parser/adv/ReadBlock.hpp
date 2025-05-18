/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/adv/RequireToken.hpp>
#include <slu/parser/basic/CharInfo.hpp>

namespace slu::parse
{
	//startCh == in.peek() !!!
	inline bool isBasicBlockEnding(AnyInput auto& in, const char startCh)
	{
		if constexpr (in.settings() & sluSyn)
		{
			if (startCh == '}') return true;
			if (startCh == 'u')
			{
				if (checkTextToken(in, "until"))
					return true;
			}
			else if (startCh == 'e')
			{
				if (checkTextToken(in, "else"))
					return true;

			}
			return false;
		}

		if (startCh == 'u')
		{
			if (checkTextToken(in, "until"))
				return true;
		}
		else if (startCh == 'e')
		{
			const char ch1 = in.peekAt(1);
			if (ch1 == 'n')
			{
				if (checkTextToken(in, "end"))
					return true;
			}
			else if (ch1 == 'l')
			{
				if (checkTextToken(in, "else") || checkTextToken(in, "elseif"))
					return true;
			}
		}
		return false;
	}

	enum class SemicolMode
	{
		NONE, REQUIRE, REQUIRE_OR_KW
	};
	template<SemicolMode semicolReq, AnyInput In>
	inline bool readReturn(In& in, const bool allowVarArg)
	{
		if (checkReadTextToken(in, "return"))
		{
			in.genData.scopeReturn();

			skipSpace(in);

			const char ch1 = in.peek();

			if (ch1 == ';')
				in.skip();//thats it
			else if (isBasicBlockEnding(in, ch1))
			{
				if constexpr (semicolReq == SemicolMode::REQUIRE)
					requireToken(in, ";");//Lazy way to throw error, maybe fix later?
			}
			else
			{
				in.genData.scopeReturn(readExpList(in, allowVarArg));

				if constexpr (semicolReq != SemicolMode::NONE)
					requireToken(in, ";");
				else
					readOptToken(in, ";");
			}
			return true;
		}
		return false;
	}

	template<bool isLoop, AnyInput In>
	inline Block<In> readBlock(In& in, const bool allowVarArg)
	{
		/*
			block ::= {stat} [retstat]
			retstat ::= return [explist] [‘;’]
		*/

		skipSpace(in);

		in.genData.pushAnonScope(in.getLoc());

		while (true)
		{
			skipSpace(in);

			if (!in)//File ended, so block ended too
				break;

			const char ch = in.peek();

			if (ch == 'r')
			{
				if (readReturn<SemicolMode::NONE>(in, allowVarArg))
					break;// no more loop
			}
			else if (isBasicBlockEnding(in, ch))
				break;// no more loop

			// Not some end / return keyword, must be a statement

			readStatement<isLoop>(in, allowVarArg);
		}
		return in.genData.popScope(in.getLoc());
	}

	template<bool isLoop, AnyInput In>
	inline Block<In> readBlockNoStartCheck(In& in, const bool allowVarArg)
	{
		Block<In> bl = readBlock<isLoop>(in, allowVarArg);
		requireToken(in, sel<In>("end", "}"));

		return bl;
	}
}
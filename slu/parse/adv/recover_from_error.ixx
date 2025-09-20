module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <cstdint>
#include <unordered_set>

export module slu.parse.adv.recover_from_error;

import slu.char_info;
import slu.ast.state;
import slu.parse.input;
import slu.parse.com.skip_space;
import slu.parse.com.str;
import slu.parse.com.tok;

namespace slu::parse
{
	inline bool trySkipMultilineString(AnyInput auto& in)
	{
		const char ch1 = in.peekAt(1);
		if (ch1 == '=' || ch1 == '[')
		{
			readStringLiteral(in, '[');
			return true;
		}
		return false;
	}

	export template<size_t TOK_SIZE>
	bool recoverErrorTextToken(AnyInput auto& in, const char(&tok)[TOK_SIZE])
	{
		while (in)
		{
			const char ch = in.peek();
			switch (ch)
			{
			case ' ':
			case '\n':
			case '\r':
			case '\t':
			case '\f':
			case '\v':
			case '-':
			{
				skipSpace(in);

				do
				{
					const char chl = in.peek();
					switch (chl)
					{
					case '"':
					case '\'':
						readStringLiteral(in, chl);
						break;
					case '[':
						if (!trySkipMultilineString(in))
							goto break_loop;
						break;
					default:
						goto break_loop;
					}
				} while (skipSpace(in) || slu::isStrStarter(in.peek()));
			break_loop:
				break;
			}
			case '"':
			case '\'':
				readStringLiteral(in, ch);
				break;
			case '[':
				trySkipMultilineString(in);
				break;
			default:
				break;
			}

			if (checkTextToken(in, tok))
			{// Found it, recovered!
				return true;
			}
			in.skip();//Not found, try at next char
		}
		return false;
	}
}
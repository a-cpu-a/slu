/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

#include <slu/paint/SemOutputStream.hpp>

namespace slu::paint
{
	// You are likely not using srgb for your colors.
	// (using srgb colors means that they get brightened by the gpu/whatever)
	// (so we need to darken them to compensate)
	template<bool SRGB>
	struct StyledColorConverter
	{
		static constexpr float pow2_2(float x) 
		{
			// Use a safe initial guess in (0, 1), like 1.0f for all small x
			float y = 1.0f;
			
			// Newton-Raphson to solve y^5 = x
			for (int i = 0; i < 6; ++i)
			{
				float y4 = y * y * y * y;
				y = (4.0f * y + x / y4) / 5.0f;
			}

			return x * x * y;
		}

		static constexpr uint32_t col(uint32_t color)
		{
			if (SRGB)
			{
				//convert to sRGB, using pow(x,2.2)
				const float r = ((color >> 16) & 0xFF) / 255.0f;
				const float g = ((color >> 8) & 0xFF) / 255.0f;
				const float b = ((color) & 0xFF) / 255.0f;
				
				color = ((uint32_t)(pow2_2(r) * 255.5f) << 16) |
						((uint32_t)(pow2_2(g) * 255.5f) << 8) |
					((uint32_t)(pow2_2(b) * 255.5f));
			}

			return (color & 0x00FFFFFF) | 0xFF000000;
		}

		static constexpr uint32_t tok2ColBuild(const Tok tok)
		{
			switch (tok)
			{
				//Add colors:
			case Tok::GEN_OP: return col(0x8E98A8);
			case Tok::PUNCTUATION: return col(0xA9B0BC);
			case Tok::BRACES:
			case Tok::MP_IDX:
			case Tok::MP: return col(0xBFC4CC);

			case Tok::NAME: return col(0xD8D8D8);
			case Tok::NAME_TABLE: return col(0x9CA5B2);
			case Tok::NAME_TYPE: return col(0x36A57E);
			case Tok::NAME_LIFETIME: return col(0x829167);
			case Tok::NAME_LABEL: return col(0x72CC82);
			case Tok::BUILITIN_VAR: return col(0x6582C1);

			case Tok::NUMBER: return col(0x539BBD);
			case Tok::NUMBER_KIND: return col(0x6B73A1);
			case Tok::NUMBER_TYPE: return col(0x90BBCC);

			case Tok::STRING: return col(0x7AA737);
			case Tok::STRING_OUT: return col(0xACD163);

			case Tok::IN:
			case Tok::COND_STAT: return col(0xBD71C4);
			case Tok::END_STAT:
			case Tok::FN_STAT: return col(0xCE4C4C);
			case Tok::VAR_STAT: return col(0xD87D7D);
			case Tok::ASSIGN: return col(0xE5506F);

			case Tok::COMMENT_WIP: return col(0xCCFF00);
			case Tok::WHITESPACE: return col(0x575763);
			case Tok::COMMENT_OUTER: return col(0x3D3D44);
			case Tok::COMMENT_DOC: return col(0x378731);
			case Tok::COMMENT_DOC_OUTER: return col(0x196022);


			case Tok::AS:
			case Tok::CON_STAT: return col(0xD8B770);
			case Tok::PAT_RESTRICT: return col(0xBFC4CC);
			case Tok::DROP_STAT: return col(0x666CCC);
			case Tok::EX_TINT: return col(0x22275B);
			case Tok::COMP_TINT: return col(0xC6C611);
			case Tok::ANNOTATION: return col(0xA8A820);
			case Tok::SPLICER: return col(0xFF00FF);//TODO: maybe some yellow?
			case Tok::SPLICE_VAR: return col(0xFF00FF);//TODO: maybe some yellow? or maybe cyan/light blue

			case Tok::PTR_CONST:
			case Tok::REF: return col(0x8AB6CE);

			case Tok::MUT: return col(0xFF9647);
			case Tok::REF_SHARE: return col(0xE4C4FF);
			case Tok::RANGE: return col(0x498CFF);
			case Tok::DEREF: return col(0xBFC4CC);
			case Tok::NOT: return col(0xFF4242);
			case Tok::ARRAY_CONSTRUCT: return col(0xA6BC6D);
			case Tok::TRY: return col(0x68997E);
			case Tok::DYN: return col(0xBF87FF);
			case Tok::IMPL: return col(0xCE905A);
				
			case Tok::AND: return col(0x5787DB);
			case Tok::OR: return col(0xB58055);
			case Tok::MOD: return col(0xB565AB);
			case Tok::ADD: return col(0xAAFF71);
			case Tok::SUB: return col(0xFF4242);
			case Tok::MUL: return col(0xFFFF71);
			case Tok::DIV: return col(0xFF60E4);
			case Tok::EXP: return col(0xFFFFFF);
			case Tok::BIT_AND: return col(0x74C2FF);
			case Tok::BIT_OR: return col(0x4CFFDA);
			case Tok::BIT_XOR: return col(0xFFAA72);
			case Tok::CONCAT: return col(0x899EC4);
			case Tok::SIZEOF: return col(0xA8A820);
			case Tok::NEG: return col(0xFF4242);
			case Tok::BITNOT: return col(0xFF6060);

			case Tok::SHL:
			case Tok::SHR: return col(0xFFC6FF);

			default:
				return 0xFFFF00FF;//error, magenta
				//return (0x122311 * uint32_t(tok) ^ 0x388831 * (uint32_t(tok) + 1)) | 0xFF000000;
			}
		}
		constexpr static uint32_t ac = col(0xE5506F);
		static_assert(tok2ColBuild(Tok::ASSIGN)== ac);
		static_assert(SRGB || ((0xE5506F | 0xFF000000) == ac));

		static constexpr auto buildColors()
		{
			std::array<uint32_t, size_t(Tok::ENUM_COUNT)> colors{};
			for (size_t i = 0; i < colors.size(); ++i)
				colors[i] = tok2ColBuild(Tok(i));
			return colors;
		}

		static constexpr auto COLORS = buildColors();
		static constexpr uint32_t tok2Col(const Tok tok) {
			return COLORS[size_t(tok)];
		}

		static constexpr uint32_t from(const Tok tok, const Tok overlayTok) {
			return basicTokBlend<StyledColorConverter>(tok, overlayTok);
		}
	};
}
/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/Settings.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/State.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/paint/SemOutputStream.hpp>
#include <slua/paint/Paint.hpp>

namespace slua::paint
{
	inline std::string getCssFor(const AnySemOutput auto& se) {
		
		std::unordered_set<uint32_t> colors;

		for (const auto& chList : se.out)
		{
			for (const uint32_t chCol : chList)
			{
				colors.insert(chCol & 0xFFFFFF);
			}
		}
		std::string res;
		for (const uint32_t col : colors)
		{
			res += ".C";
			for (size_t i = 0; i < 6; i++)
			{
				const uint8_t nibble = (col >> (20 - i * 4)) & 0xF;
				if (nibble < 10)
					res += '0' + nibble;
				else
					res += 'A' + (nibble - 10);
			}
			res += "{color:#";
			for (size_t i = 0; i < 6; i++)
			{
				const uint8_t nibble = (col >> (20 - i * 4)) & 0xF;
				if (nibble < 10)
					res += '0' + nibble;
				else
					res += 'A' + (nibble - 10);
			}
			res += '}';
		}
		return res;
	}
	/*
	Make sure to reset in first: `in.reset();`
	*/
	inline std::string toHtml(AnySemOutput auto& se,const bool includeStyle) {
		std::string res;
		if (includeStyle)
		{
			res += "<style>" + getCssFor(se) + "</style>";
		}
		res += "<code class=slua-code-box><span>";
		uint32_t prevCol = 0xFFFFFF;

		parse::ParseNewlineState nlState = parse::ParseNewlineState::NONE;
		while (se.in)
		{
			const parse::Position loc = se.in.getLoc();
			const char ch = se.in.get();

			//Already skipping
			if (parse::manageNewlineState<false>(ch, nlState, se.in))
			{
				res += "<br>";
				continue;
			}


			const auto& lineCols = se.out[loc.line - 1];
			const uint32_t col = (lineCols.size()<=loc.index ? se.commentPair() : lineCols[loc.index]) & 0xFFFFFF;

			if (prevCol != col)
			{
				prevCol = col;
				res += "</span><span class=C";
				for (size_t i = 0; i < 6; i++)
				{
					const uint8_t nibble = (col >> (20 - i * 4)) & 0xF;
					if (nibble < 10)
						res += '0' + nibble;
					else
						res += 'A' + (nibble - 10);
				}
				res += ">";
			}

			if (ch == '\'')
			{
				res += "&#39;";
				continue;
			}
			if (ch == '\"')
			{
				res += "&#34;";
				continue;
			}
			if (ch == '>')
			{
				res += "&gt;";
				continue;
			}
			if (ch == '<')
			{
				res += "&lt;";
				continue;
			}
			if (ch == '&')
			{
				res += "&amp;";
				continue;
			}
			res += ch;
		}
		res += "</span></code>";
		return res;
	}
}
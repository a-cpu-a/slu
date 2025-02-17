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


namespace sluaParse
{
	//Here, so streamed outputs can be made
	template<class T>
	concept AnyOutput = requires(T t) {
		{ t.add(char(1)) } -> std::same_as<T&>;
		{ t.add(std::string_view()) } -> std::same_as<T&>;
		{ t.add(std::span<const uint8_t>()) } -> std::same_as<T&>;

		{ t.newLine() } -> std::same_as<T&>;

		{ t.indent() } -> std::same_as<T&>;
		{ t.tabUp() } -> std::same_as<T&>;
		{ t.unTab() } -> std::same_as<T&>;
	};

	struct Output
	{
		std::vector<uint8_t> text;
		uint64_t tabs;

		Output& add(const char ch) 
		{
			text.push_back(ch);
			return *this;
		}
		Output& add(const std::string_view sv) 
		{
			text.insert(text.end(), sv.begin(), sv.end());
			return *this;
		}
		Output& add(std::span<const uint8_t> sp) 
		{
			text.insert(text.end(), sp.begin(), sp.end());
			return *this;
		}

		Output& indent() 
		{
			text.insert(text.end(), tabs, '\t');
			return *this;
		}
		Output& tabUp()
		{
			tabs++;
			return *this;
		}
		Output& unTab() 
		{
			tabs--;
			return *this;
		}

		Output& newLine()
		{
			text.push_back('\n');
			indent();
			return *this;
		}

	};
}
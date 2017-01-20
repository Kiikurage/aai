var moveCount = 0;
(function(O) {
    'use strict';

    // UI {{{1

    function drawGameBoard(board, player, moves) {
        var ss = [];
        var attackable = [];
        moves.forEach(function(m) {
            if (!m.isPassingMove)
                attackable[O.ix(m.x, m.y)] = true;
        });

        ss.push('<table>');
        for (var y = -1; y < O.N; y++) {
            ss.push('<tr>');
            for (var x = -1; x < O.N; x++) {
                if (0 <= y && 0 <= x) {
                    ss.push('<td class="');
                    ss.push('cell');
                    ss.push(' ');
                    ss.push(attackable[O.ix(x, y)] ? player : board[O.ix(x, y)]);
                    ss.push(' ');
                    ss.push(attackable[O.ix(x, y)] ? 'attackable' : '');
                    ss.push('" id="');
                    ss.push('cell_' + x + '_' + y);
                    ss.push('">');
                    ss.push('<span class="disc"></span>');
                    ss.push('</td>');
                } else if (0 <= x && y === -1) {
                    ss.push('<th>' + String.fromCharCode('a'.charCodeAt(0) + x) + '</th>');
                } else if (x === -1 && 0 <= y) {
                    ss.push('<th>' + (y + 1) + '</th>');
                } else /* if (x === -1 && y === -1) */ {
                    ss.push('<th></th>');
                }
            }
            ss.push('</tr>');
        }
        ss.push('</table>');

        $('#game-board').html(ss.join(''));
        $('#current-player-name').text(player);
    }

    function resetUI() {
        $('#console').empty();
        $('#message').empty();
    }

    function setUpUIToChooseMove(gameTree) {
        $('#message').text('choose your move');
        gameTree.moves.forEach(function(m, i) {
            if (m.isPassingMove) {
                $('#console').append(
                    $('<input type="button" class="btn">')
                    .val(O.nameMove(m))
                    .click(function() {
                        shiftToNewGameTree(O.force(m.gameTreePromise));
                    })
                );
            } else {
                $('#cell_' + m.x + '_' + m.y)
                    .click(function() {
                        shiftToNewGameTree(O.force(m.gameTreePromise));
                    });
            }
        });
    }

    function setUpUIToReset() {
        resetGame();
        if ($('#repeat-games:checked').length)
            startNewGame();
    }

    var minimumDelayForAI = 500; // milliseconds

    function chooseMoveByAI(gameTree, ai) {
        $('#message').text('Now thinking...');
        setTimeout(
            function () {
                var start = Date.now();
                var newGameTree = O.force(ai.findTheBestMove(gameTree).gameTreePromise);
                var end = Date.now();
                var delta = end - start;
                setTimeout(
                    function () {
                        shiftToNewGameTree(newGameTree);
                    },
                    Math.max(minimumDelayForAI - delta, 1)
                );
            },
            1
        );
    }


    function chooseMoveByMyAI(gameTree, playerType) {
        $('#message').text('Now thinking...');
        moveCount += 1;
        gameTree.count = moveCount;
        gameTree.aiType = playerType;
        console.log(playerType);
        $.ajax({
                url: '/getMove',
                type: 'POST',
                data: JSON.stringify(gameTree),
                dataType: 'json',
                contentType: 'application/json'
            })
            .then(function(data, textStatus, jqXHR) {
                console.log(data);

                var start = Date.now();
                if (data["index"] > -1) {
                    var newGameTree = O.force(gameTree.moves[data["index"]].gameTreePromise);
                } else {
                    console.log("sended data is wrong")
                }

                var end = Date.now();
                var delta = end - start;
                shiftToNewGameTree(newGameTree);
            })
            .fail(function(jqXHR, textStatus, errorThrown) {
                console.log("failed");
            });
    }

    function showWinner(board) {
        var r = O.judge(board);
        var n = {};
        var EMPTY = 'empty';
        var WHITE = 'white';
        var BLACK = 'black';
        n[BLACK] = 0;
        n[WHITE] = 0;
        n[EMPTY] = 0;
        for (var i = 0; i < board.length; i++)
            n[board[i]]++;
        var numText = Math.abs(n[BLACK]-n[WHITE]);

        if (r == 0) {
            $('#message').css("fontSize", "35px");
            $('#message').css("color", "#FF5733");
            $('#message').text("DRAW  GAME");
        } else {

            $('#message').css("fontSize", "20px");
            $('#message').html('The winner is ' + (r === 1 ? O.BLACK : O.WHITE) +'<br><br>'+'Difference is '+ numText);
        }

    }

    var playerTable = {};

    function makePlayer(playerType) {
        if (playerType === 'human') {
            return setUpUIToChooseMove;
        } else if (playerType.match('policy')) {
            console.log('myAI');
            return function(gameTree) {
                chooseMoveByMyAI(gameTree, playerType);
            };
        } else {
            console.log('OriginalAI');
            var ai = O.makeAI(playerType);
            return function(gameTree) {
                chooseMoveByAI(gameTree, ai);
            };
        }
    }

    function blackPlayerType() {
        return $('#black-player-type').val();
    }

    function whitePlayerType() {
        return $('#white-player-type').val();
    }

    function swapPlayerTypes() {
        var t = $('#black-player-type').val();
        $('#black-player-type').val($('#white-player-type').val()).change();
        $('#white-player-type').val(t).change();
    }

    function judgeFinishGame(gameTree) {
        var board = gameTree.board;
        var numStone = 0;
        for (var i = 0; i < board.length; i++) {
            if (board[i] != "empty")
                numStone += 1;
        }
        if (numStone == 64) {
            return true;
        } else {
            return false;
        }

    }

    function shiftToNewGameTree(gameTree) {
        drawGameBoard(gameTree.board, gameTree.player, gameTree.moves);
        resetUI();
        if (judgeFinishGame(gameTree)) {
            showWinner(gameTree.board);
            setUpUIToReset();
        } else {
            if (gameTree.moves.length === 0) {
                showWinner(gameTree.board);
                recordStat(gameTree.board);
                if ($('#repeat-games:checked').length)
                    showStat();
                setUpUIToReset();
            } else {
                playerTable[gameTree.player](gameTree);
            }
        }


    }

    var stats = {};

    function recordStat(board) {
        var s = stats[[blackPlayerType(), whitePlayerType()]] || {
            b: 0,
            w: 0,
            d: 0
        };
        var r = O.judge(board);
        if (r === 1)
            s.b++;
        if (r === 0)
            s.d++;
        if (r === -1)
            s.w++;
        stats[[blackPlayerType(), whitePlayerType()]] = s;
    }

    function showStat() {
        var s = stats[[blackPlayerType(), whitePlayerType()]];
        $('#stats').text('Black: ' + s.b + ', White: ' + s.w + ', Draw: ' + s.d);
    }

    function resetGame() {
        $('#preference-pane :input:not(#repeat-games)')
            .removeClass('disabled')
            .removeAttr('disabled');
    }

    function startNewGame() {
        $('#message').css("fontSize", "14px");
        $('#message').css("color", "black");

        $('#preference-pane :input:not(#repeat-games)')
            .addClass('disabled')
            .attr('disabled', 'disabled');
        playerTable[O.BLACK] = makePlayer(blackPlayerType());
        playerTable[O.WHITE] = makePlayer(whitePlayerType());

        shiftToNewGameTree(O.makeInitialGameTree());
    }




    // Startup {{{1

    $('#start-button').click(function() {
        startNewGame();
    });
    $('#add-new-ai-button').click(function() {
        O.addNewAI();
    });
    $('#swap-player-types-button').click(function() {
        swapPlayerTypes();
    });
    resetGame();
    drawGameBoard(O.makeInitialGameBoard(), '-', []);




    //}}}
})(othello);
// vim: expandtab softtabstop=2 shiftwidth=2 foldmethod=marker
